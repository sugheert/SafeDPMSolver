# Rasterized Environment Encoder — Implementation Harness

## Overview

This harness describes the end-to-end workflow for replacing the parametric
`EllipsoidFiLMEncoder` in `score_net_ellipsoids.py` with a rasterized occupancy
map and a CNN-based encoder. The goal is to make all scene geometry — walls,
maze boundaries, and interior obstacles — visible to the denoising U-Net, fixing
the trajectory-through-obstacle failure mode.

---

## Requirements

### Python Packages

```
torch >= 2.0
torchvision >= 0.15       # for ResNet-18 backbone
einops >= 0.6
numpy >= 1.24
Pillow >= 9.0             # for bitmap rendering / debug visualization
```

Install with:

```bash
pip install torch torchvision einops numpy Pillow
```

### Hardware

| Role | Minimum | Recommended |
|---|---|---|
| Training | 1× GPU, 16 GB VRAM | 2× RTX A6000 |
| Inference | 1× GPU, 8 GB VRAM | Any modern GPU |

### File Dependencies

```
score_net_ellipsoids.py        ← base file being modified
train_large_ellipsoids_diffuser.ipynb  ← training loop (update forward call here)
```

---

## Architecture Changes at a Glance

```
BEFORE
──────
ellipsoids [B, 5, 4]
    └─► EllipsoidFiLMEncoder (DeepSet)
            └─► global vector [B, 256]
                    └─► FiLM inject into every ResidualTemporalBlock

AFTER
─────
occ_map [B, 1, H, W]   ← rasterized walls + ellipsoids
    ├─► MapEncoder (CNN)
    │       └─► feature map [B, C, H', W']
    │               ├─► GlobalPoolHead
    │               │       └─► global vector [B, 256]  → FiLM (coarse scene)
    │               └─► F.grid_sample at waypoint (x,y)
    │                       └─► local features [B, T, C] → concat to waypoint input
    └─► (reused across all denoising steps — computed once per scene)
```

---

## Step 1 — Rasterize the Scene

Convert all geometry into a single `[B, 1, H, W]` binary occupancy bitmap at
dataset generation time (or on-the-fly during training). Both walls and interior
ellipsoids are rasterized into the same bitmap so nothing is hidden from the model.

```python
# rasterize.py

import torch

def rasterize_scene(
    ellipsoids: torch.Tensor,   # [B, N, 4]  (cx, cy, a, b) in [-1, 1] space
    wall_bitmap: torch.Tensor,  # [B, 1, H, W]  pre-rendered wall occupancy
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:              # [B, 1, H, W]  combined occupancy in {0, 1}
    """
    Rasterizes ellipsoid obstacles onto a pixel grid and merges with the
    wall bitmap. Coordinates are assumed to be in the [-1, 1] range.

    Args:
        ellipsoids  : Ellipsoid parameters. Pad absent ellipsoids with zeros.
        wall_bitmap : Binary bitmap of maze walls, same spatial resolution.
        H, W        : Output resolution (should match wall_bitmap).

    Returns:
        Binary occupancy map combining walls and all ellipsoids.
    """
    B = ellipsoids.shape[0]
    device = ellipsoids.device

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    grid_x = grid_x[None, None]   # [1, 1, H, W]
    grid_y = grid_y[None, None]

    occ = torch.zeros(B, 1, H, W, device=device)

    for i in range(ellipsoids.shape[1]):
        cx = ellipsoids[:, i, 0].view(B, 1, 1, 1)
        cy = ellipsoids[:, i, 1].view(B, 1, 1, 1)
        a  = ellipsoids[:, i, 2].view(B, 1, 1, 1)
        b  = ellipsoids[:, i, 3].view(B, 1, 1, 1)

        # Skip zero-padded ellipsoids (a == 0 means absent)
        valid = (a > 0).float()

        inside = (((grid_x - cx) / (a + 1e-6)) ** 2
                + ((grid_y - cy) / (b + 1e-6)) ** 2) <= 1.0
        occ = occ + valid * inside.float()

    return (occ + wall_bitmap).clamp(0.0, 1.0)
```

**Important:** Pre-render wall bitmaps once and store them alongside your
dataset. Do not recompute per training step.

---

## Step 2 — CNN Map Encoder

Replace `EllipsoidFiLMEncoder` with `MapEncoder`. The encoder produces a
spatial feature map that is used for both global FiLM conditioning and
local per-waypoint sampling.

```python
# Add to score_net_ellipsoids.py

import torchvision.models as tvm
import torch.nn.functional as F

class MapEncoder(nn.Module):
    """
    Encodes a binary occupancy bitmap into a spatial feature map.

    Uses the first two layer groups of a ResNet-18 backbone (conv1 through
    layer2) adapted for single-channel input, followed by a 1×1 projection.

    Input  : [B, 1, H, W]           binary occupancy map
    Output : [B, local_dim, H/8, W/8]   spatial feature map

    For H=W=64 input, output spatial size is 8×8.
    The feature map is computed once per scene and reused across all
    denoising iterations.
    """
    def __init__(self, local_dim: int = 16):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # [B, 64,  H/4,  W/4]
            resnet.layer2,   # [B, 128, H/8,  W/8]
        )
        self.proj = nn.Conv2d(128, local_dim, kernel_size=1)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, 1, H, W]
        return self.proj(self.backbone(E))   # [B, local_dim, H/8, W/8]


class GlobalPoolHead(nn.Module):
    """
    Pools a spatial feature map into a single global conditioning vector.

    Input  : [B, local_dim, H', W']
    Output : [B, global_dim]
    """
    def __init__(self, local_dim: int = 16, global_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(local_dim, global_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(feat_map))   # [B, global_dim]


def sample_local_features(
    feat_map: torch.Tensor,    # [B, C, H', W']
    positions: torch.Tensor,   # [B, T, 2]  in [-1, 1]
) -> torch.Tensor:             # [B, T, C]
    """
    Samples the spatial feature map at each waypoint position via bilinear
    interpolation. Out-of-bounds positions return zero vectors.

    Args:
        feat_map  : Encoded occupancy feature map.
        positions : (x, y) coordinates of each waypoint, in [-1, 1].

    Returns:
        Per-waypoint local features, shape [B, T, C].
    """
    grid = positions.unsqueeze(2)                        # [B, T, 1, 2]
    feats = F.grid_sample(
        feat_map, grid,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros',
    )                                                    # [B, C, T, 1]
    return feats.squeeze(-1).permute(0, 2, 1)           # [B, T, C]
```

---

## Step 3 — Modify `TemporalUnet`

### `__init__` changes

```python
# Remove:
self.ellipsoid_encoder = EllipsoidFiLMEncoder(...)

# Add:
self.map_encoder   = MapEncoder(local_dim=local_dim)
self.global_head   = GlobalPoolHead(local_dim=local_dim, global_dim=global_dim)

# Update cond_dim (local features are concatenated to input, not to c_emb):
cond_dim = time_emb_dim + conditioning_embed_dim + global_dim

# Update U-Net input channels to accept local features per waypoint:
# state_dim stays 2 for positions; local features concatenated before first down block
dims = [state_dim + local_dim, *[unet_input_dim * m for m in dim_mults]]
```

### `forward` signature change

```python
def forward(
    self,
    x:        torch.Tensor,   # [B, T, 2]       noisy trajectory
    sigma:    torch.Tensor,   # [B]              noise level
    x_start:  torch.Tensor,   # [B, 2]           start condition
    x_goal:   torch.Tensor,   # [B, 2]           goal condition
    occ_map:  torch.Tensor,   # [B, 1, H, W]     rasterized occupancy map
) -> torch.Tensor:
    """Returns predicted noise eps : [B, T, 2]."""

    # Global conditioning
    t_emb      = self.time_mlp(sigma)
    ctx        = self.context_proj(torch.cat([x_start, x_goal], dim=-1))
    feat_map   = self.map_encoder(occ_map)           # [B, C, H', W']
    map_global = self.global_head(feat_map)          # [B, global_dim]
    c_emb      = torch.cat([t_emb, ctx, map_global], dim=-1)

    # Local conditioning — sample feature map at each waypoint position
    positions  = x[..., :2]                          # [B, T, 2]
    local_feat = sample_local_features(feat_map, positions)   # [B, T, C]

    # Concatenate local features to waypoint input before U-Net
    x_in = torch.cat([x, local_feat], dim=-1)        # [B, T, 2+C]
    x_in = rearrange(x_in, 'b t d -> b d t')        # [B, 2+C, T]

    # U-Net forward (unchanged from here)
    skips = []
    for resnet1, resnet2, downsample in self.downs:
        x_in = resnet1(x_in, c_emb)
        x_in = resnet2(x_in, c_emb)
        skips.append(x_in)
        x_in = downsample(x_in)

    x_in = self.mid_block1(x_in, c_emb)
    x_in = self.mid_block2(x_in, c_emb)

    for resnet1, resnet2, upsample in self.ups:
        x_in = torch.cat([x_in, skips.pop()], dim=1)
        x_in = resnet1(x_in, c_emb)
        x_in = resnet2(x_in, c_emb)
        x_in = upsample(x_in)

    x_in = self.final_conv(x_in)                     # [B, 2, T]
    return rearrange(x_in, 'b d t -> b t d')         # [B, T, 2]
```

---

## Step 4 — Update the Training Loop

In `train_large_ellipsoids_diffuser.ipynb`, replace the ellipsoid forward call:

```python
# BEFORE
score = model(x_noisy, sigma, x_start, x_goal, ellipsoids)

# AFTER — rasterize once per batch, then forward
occ_map = rasterize_scene(ellipsoids, wall_bitmap, H=64, W=64)
score   = model(x_noisy, sigma, x_start, x_goal, occ_map)
```

Pre-compute and cache `wall_bitmap` per environment at dataset load time, not
inside the training step, to avoid redundant rendering overhead.

---

## Step 5 — Classifier-Free Guidance (CFG)

For CFG the null (unconditional) input is a **blank occupancy map** rather than
zero ellipsoids:

```python
# Conditional forward
score_cond   = model(x, sigma, x_start, x_goal, occ_map)

# Unconditional forward — pass all-zeros bitmap
null_map     = torch.zeros_like(occ_map)
score_uncond = model(x, sigma, x_start, x_goal, null_map)

# CFG combination
score = score_uncond + guidance_scale * (score_cond - score_uncond)
```

---

## Inference Notes

- Compute `feat_map = map_encoder(occ_map)` **once** before the denoising loop
  and pass it directly into subsequent iterations — the map does not change
  across denoising steps.
- For warm-start replanning (online control), the cached `feat_map` can be
  reused across replanning frames as long as the environment has not changed,
  reducing per-step cost to a single U-Net forward pass.

---

## Expected Failure Modes Fixed

| Symptom | Root Cause | Resolution |
|---|---|---|
| Trajectories pass through walls | Walls absent from conditioning | Walls rasterized into bitmap |
| Trajectories clip interior obstacles | Global FiLM too coarse | Local per-waypoint sampling |
| Fails with complex obstacle layouts | Hard cap of 5 ellipsoids | Bitmap has no capacity limit |
| Poor OOD generalization | Parametric repr. not flexible | CNN generalizes to any geometry |

---

## Hyperparameter Reference

| Parameter | Default | Notes |
|---|---|---|
| `H`, `W` (bitmap resolution) | 64 | Match paper; increase for fine geometry |
| `local_dim` (feature channels) | 16 | Trades off memory vs. expressivity |
| `global_dim` (FiLM vector) | 256 | Keep equal to original ellipsoid output dim |
| `time_emb_dim` | 32 | Unchanged |
| `conditioning_embed_dim` | 4 | Unchanged |
| `unet_input_dim` | 32 | Unchanged |
| `dim_mults` | (1, 2, 4) | Unchanged |