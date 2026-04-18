# Rasterized Environment Encoder — Local Feature Sampling Harness

## Overview

This harness describes the end-to-end workflow for implementing **Local Feature Sampling** to condition the denoising U-Net on the environment map, matching the methodology in the "Joint Localization and Planning using Diffusion" paper.

The previous attempt to use a global FiLM vector for the map failed because compressing a 2D spatial layout into a single vector destroys the geometric alignment between the robot's trajectory and the walls. The correct approach is to encode the map into a spatial feature tensor, and then sample that tensor at the exact `(x, y)` coordinates of the noisy trajectory waypoints.

---

## Requirements

### Python Packages

```text
torch >= 2.0
torchvision >= 0.15
einops >= 0.6
numpy >= 1.24
Pillow >= 9.0
```

---

## Architecture Changes at a Glance

```text
occ_map [B, 1, H, W]   (Rasterized walls + ellipsoids)
    │
    ▼
[ MapEncoder (CNN) ]  --> Extracts Voronoi-like geometric features
    │
    ▼
feat_map [B, local_dim, H/8, W/8]
    │
    ├──► [ Bilinear Grid Sample ] ◄── Waypoint Positions (x, y) from T^(t)
    │
    ▼
local_features [B, T, local_dim]
    │
    ├──► [ Concatenate ] ◄── Waypoint State (x, y, [theta])
    │
    ▼
x_in [B, T, state_dim + local_dim]
    │
    ▼
[ Temporal U-Net ] (Conditioned globally ONLY on Goal & Time via FiLM)
```

---

## Step 1 — Rasterize the Scene

Convert all geometry into a single `[B, 1, H, W]` binary occupancy bitmap.

```python
# rasterize.py

import torch

def rasterize_scene(
    ellipsoids: torch.Tensor,   # [B, N, 4]  (cx, cy, a, b) in [-1, 1] space
    wall_bitmap: torch.Tensor,  # [B, 1, H, W]  pre-rendered wall occupancy
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:
    """
    Rasterizes ellipsoid obstacles onto a pixel grid and merges with the wall bitmap.
    """
    B = ellipsoids.shape[0]
    device = ellipsoids.device

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_x = grid_x[None, None]   # [1, 1, H, W]
    grid_y = grid_y[None, None]

    occ = torch.zeros(B, 1, H, W, device=device)

    for i in range(ellipsoids.shape[1]):
        cx = ellipsoids[:, i, 0].view(B, 1, 1, 1)
        cy = ellipsoids[:, i, 1].view(B, 1, 1, 1)
        a  = ellipsoids[:, i, 2].view(B, 1, 1, 1)
        b  = ellipsoids[:, i, 3].view(B, 1, 1, 1)

        valid = (a > 0).float()

        inside = (((grid_x - cx) / (a + 1e-6)) ** 2
                + ((grid_y - cy) / (b + 1e-6)) ** 2) <= 1.0
        occ = occ + valid * inside.float()

    return (occ + wall_bitmap).clamp(0.0, 1.0)
```

---

## Step 2 — CNN Map Encoder & Local Sampler

Replace the old `EllipsoidFiLMEncoder` and `GlobalPoolHead`. We only need the spatial feature map and a function to sample it at specific waypoints.

```python
# Add to score_net_ellipsoids.py

import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F
import torch

class MapEncoder(nn.Module):
    """
    Encodes a binary occupancy bitmap into a spatial feature map.
    Uses a modified ResNet-18 backbone.
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


def sample_local_features(
    feat_map: torch.Tensor,    # [B, C, H', W']
    positions: torch.Tensor,   # [B, T, 2]  in [-1, 1]
) -> torch.Tensor:             # [B, T, C]
    """
    Samples the spatial feature map at each waypoint position via bilinear interpolation.
    """
    # F.grid_sample expects grid shape [B, H_out, W_out, 2]
    # We treat T as W_out and set H_out=1
    grid = positions.unsqueeze(1)                        # [B, 1, T, 2]
    feats = F.grid_sample(
        feat_map, grid,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros',
    )                                                    # [B, C, 1, T]
    return feats.squeeze(2).permute(0, 2, 1)             # [B, T, C]
```

---

## Step 3 — Modify `TemporalUnet`

Provide the model with the ability to take the local features, but remove map data from the global FiLM embeddings.

### `__init__` changes

```python
# Add to __init__:
self.map_encoder = MapEncoder(local_dim=local_dim)

# Update cond_dim (Map is NO LONGER part of the global FiLM vector):
cond_dim = time_emb_dim + conditioning_embed_dim 

# Update U-Net input channels to accept local features per waypoint:
dims = [state_dim + local_dim, *[unet_input_dim * m for m in dim_mults]]
```

### `forward` method

```python
from einops import rearrange

def forward(
    self,
    x:        torch.Tensor,   # [B, T, 2]       noisy trajectory
    sigma:    torch.Tensor,   # [B]             noise level
    x_start:  torch.Tensor,   # [B, 2]          start condition
    x_goal:   torch.Tensor,   # [B, 2]          goal condition
    occ_map:  torch.Tensor,   # [B, 1, H, W]    rasterized occupancy map
    cached_feat_map = None    # [B, C, H', W']  Optional pre-computed map for inference
) -> torch.Tensor:
    
    # 1. Global conditioning (Time & Goal ONLY)
    t_emb      = self.time_mlp(sigma)
    ctx        = self.context_proj(torch.cat([x_start, x_goal], dim=-1))
    c_emb      = torch.cat([t_emb, ctx], dim=-1)

    # 2. Extract or load Map Features
    if cached_feat_map is not None:
        feat_map = cached_feat_map
    else:
        feat_map = self.map_encoder(occ_map)         # [B, C, H', W']

    # 3. Local conditioning — sample feature map at each waypoint position
    positions  = x[..., :2]                          # [B, T, 2]
    local_feat = sample_local_features(feat_map, positions)   # [B, T, C]

    # 4. Concatenate local features to waypoint input BEFORE U-Net convolutions
    x_in = torch.cat([x, local_feat], dim=-1)        # [B, T, 2+C]
    x_in = rearrange(x_in, 'b t d -> b d t')         # [B, 2+C, T]

    # U-Net forward blocks
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

## Step 4 — Update the Training Loop & CFG

In `train_large_ellipsoids_diffuser.ipynb`, replace the ellipsoid forward call. The map rendering happens on the fly.

```python
# Inside training loop:
ells_norm = normalize_ellipsoids(ellipsoids, xy_min_t.to(device), xy_range_t.to(device))
occ_map   = rasterize_scene(ells_norm, wall_bitmap, MAP_H, MAP_W)

# Forward pass
loss, info = ve(traj, xs, xg, occ_map, p_uncond=P_UNCOND)
```

In `ve_diffusion_ellipsoids.py`, implement CFG dropout by zeroing out the occupancy map, which simulates a completely obstacle-free environment.

```python
# CFG dropout logic inside loss()
occ_map_dropped = occ_map.clone()
if drop_mask.any(): # assuming drop_mask is your unconditional probability mask
    occ_map_dropped[drop_mask] = 0.0

score = self.model(x, sigma, x_start, x_goal, occ_map_dropped)
```

---

## Step 5 — Inference Speedup (Caching)

During sampling (e.g., inside your DPM-Solver using sigma parameterization), the obstacle environment is static. Re-running `MapEncoder` at every timestep drastically slows down inference. Cache the `feat_map` beforehand.

In `samplers_ellipsoids_cfg.py`:

```python
def dpm_solver_1_cbf_cfg_sample(
    model, x, x_start, x_goal, occ_map, obstacles, ...
):
    # 1. Pre-compute and cache feature maps ONCE
    with torch.no_grad():
        cond_feat_map = model.map_encoder(occ_map)
        null_map = torch.zeros_like(occ_map)
        uncond_feat_map = model.map_encoder(null_map)

    # ... DPM-Solver step loop ...
    for i in range(len(t_steps) - 1):
        # ... calculate sigma ...

        # 2. Pass cached maps directly to the model function
        score_cond = model(x_t, sigma_t, x_start, x_goal, occ_map=None, cached_feat_map=cond_feat_map)
        score_uncond = model(x_t, sigma_t, x_start, x_goal, occ_map=None, cached_feat_map=uncond_feat_map)

        score = score_uncond + guidance_scale * (score_cond - score_uncond)
        
        # ... DPM solver update utilizing score ...
```