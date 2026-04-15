# Plan: Rasterized Environment Encoder — Implementation

## Context

The current `TemporalUnet` in `score_net_ellipsoids.py` conditions on ellipsoids via a DeepSet encoder
(`EllipsoidFiLMEncoder`), which produces a single global vector `[B, 256]`. This approach has two
failure modes: (1) maze walls are completely invisible to the model, causing trajectories to pass
through them; (2) the 5-ellipsoid cap prevents complex layouts. The fix is to replace the parametric
encoder with a CNN that reads a rasterized binary occupancy map containing **both walls and ellipsoids**,
encoded as a global FiLM conditioning vector (no local per-waypoint features).

The harness at `EllipsoidalCBFSampling/rasterized_env_harness.md` specifies the design.

---

## Architecture (Simplified — Global FiLM only, no local feature sampling)

```
occ_map [B, 1, H, W]
    → MapEncoder (CNN)          → feat_map [B, local_dim, H/8, W/8]
    → GlobalPoolHead            → map_global [B, global_dim=256]
    → cat([t_emb, ctx, map_global])  → c_emb [B, cond_dim]
    → FiLM inject into every ResidualTemporalBlock (unchanged U-Net forward)
```

No `sample_local_features`, no local feature concatenation to waypoint input.
U-Net input channels remain `[state_dim, ...]` (dims list unchanged from original).

---

## Files to Create

### `EllipsoidalCBFSampling/rasterize.py` (new)

Three utilities:

**`get_large_maze_wall_bitmap(xy_min, xy_range, H=64, W=64) → Tensor[1,1,H,W]`**
- Hard-codes `_LARGE_MAZE_MAP` (9×12 grid matching the notebook's `_MAZE_MAP`)
- Linspace grid in normalized [-1,1] space; converts to world coords:
  `gx_world = (grid_x + 1) / 2 * xy_range[0] + xy_min[0]`
- Marks pixels inside each wall cell (axis-aligned unit squares, centers at `cx=-5.5+j`, `cy=4.0-i`)
- Returns float32 binary `[1, 1, H, W]`; call once at training start and cache on GPU

**`normalize_ellipsoids(ellipsoids_world, xy_min, xy_range) → Tensor[B,N,4]`**
- Centers: `cx_norm = 2*(cx - xy_min[0]) / xy_range[0] - 1.0`  (same for cy)
- Radii: `a_norm = 2*a / xy_range[0]`  (scale only; same for b)
- Zero-padded ellipsoids remain near-zero after normalization (centers at 0,0 shift slightly; the
  `a > 0` guard in `rasterize_scene` skips them correctly)

**`rasterize_scene(ellipsoids_norm, wall_bitmap, H=64, W=64) → Tensor[B,1,H,W]`**
- Exact meshgrid + ellipse equation from harness; `a > 0` guard skips zero-padded obstacles
- Merges with `wall_bitmap` broadcast over batch: `(occ + wall_bitmap).clamp(0, 1)`

---

## Files to Modify

### `EllipsoidalCBFSampling/models/score_net_ellipsoids.py`

**Add after existing imports:**
```python
import torchvision.models as tvm
```

**Add two new classes (after `EllipsoidFiLMEncoder`, keep existing encoder for reference):**

`MapEncoder(local_dim=16)`:
- ResNet-18 backbone: `conv1` patched to 1-channel input, then `bn1+relu+maxpool+layer1+layer2`
- 1×1 projection: `Conv2d(128, local_dim, 1)`
- Output: `[B, local_dim, H/8, W/8]` (8×8 spatial for H=W=64 input)

`GlobalPoolHead(local_dim=16, global_dim=256)`:
- `AdaptiveAvgPool2d(1)` → `Flatten` → `Linear(local_dim, global_dim)` → `ReLU`
- Output: `[B, global_dim]`

**`TemporalUnet.__init__` changes:**
- New params: `local_dim: int = 16`, `global_dim: int = 256`
- Remove `ellipsoid_hidden_dim`, `ellipsoid_output_dim`, `self.ellipsoid_encoder`
- Add `self.map_encoder = MapEncoder(local_dim=local_dim)`
- Add `self.global_head = GlobalPoolHead(local_dim=local_dim, global_dim=global_dim)`
- `cond_dim = time_emb_dim + conditioning_embed_dim + global_dim`  (256 replaces ellipsoid_output_dim)
- `dims = [state_dim, *[unet_input_dim * m for m in dim_mults]]`  (unchanged — no local concat)

**`TemporalUnet.forward` signature change:**
```python
def forward(self, x, sigma, x_start, x_goal, occ_map):
```
Body changes (only the conditioning block):
```python
feat_map   = self.map_encoder(occ_map)            # [B, local_dim, H', W']
map_global = self.global_head(feat_map)           # [B, global_dim]
c_emb      = torch.cat([t_emb, ctx, map_global], dim=-1)
x = rearrange(x, 'b t d -> b d t')               # [B, 2, T]  — unchanged from here
```
U-Net down/mid/up/final blocks are unchanged.

### `EllipsoidalCBFSampling/models/ve_diffusion_ellipsoids.py`

- Rename `ellipsoids` → `occ_map` in `loss()` signature and body
- CFG dropout: `occ_map_dropped = occ_map.clone(); occ_map_dropped[drop_mask] = 0.0`
  (blank map = unconditional, analogous to zero ellipsoids)
- Pass `occ_map_dropped` to `self.model(...)`

### `EllipsoidalCBFSampling/models/samplers_ellipsoids_cfg.py`

- `dpm_solver_1_cfg_sample`: replace `ellipsoids` param with `occ_map`;
  null conditioning: `null_map = torch.zeros_like(occ_map)`
- `dpm_solver_1_cbf_cfg_sample`: same model-conditioning change;
  `obstacles` param (raw world-coord geometry for CBF) is unchanged

### `EllipsoidalCBFSampling/train_large_ellipsoids_diffuser.ipynb`

**Cell 2 (Parameters):** Add `MAP_H = MAP_W = 64`, `LOCAL_DIM = 16`, `GLOBAL_DIM = 256`;
remove `ELLIPSOID_HIDDEN_DIM`, `ELLIPSOID_OUTPUT_DIM`.

**Cell 4 (Imports):** Add rasterize imports:
```python
from EllipsoidalCBFSampling.rasterize import (
    get_large_maze_wall_bitmap, normalize_ellipsoids, rasterize_scene
)
```

**After dataset load (end of Cell 6):** Compute wall bitmap once:
```python
xy_min_t   = torch.tensor(ds.xy_min,   dtype=torch.float32)
xy_range_t = torch.tensor(ds.xy_range, dtype=torch.float32)
wall_bitmap = get_large_maze_wall_bitmap(ds.xy_min, ds.xy_range, MAP_H, MAP_W).to(device)
```

**New visualization cell (after wall_bitmap computation):**
A 1×2 matplotlib figure:
- Left panel: draw_maze + draw_ellipsoid overlaid on axes (world coords), for a fixed sample from `fix_ells`
- Right panel: `plt.imshow(occ_sample[0,0].cpu(), origin='lower', cmap='gray_r')` of the rasterized
  map for the same sample, with title showing resolution
- Shows that walls + ellipsoids are correctly encoded in the bitmap

**Cell 8 (Model Setup):**
```python
score_net = TemporalUnet(
    state_dim=2, T_steps=T_STEPS,
    unet_input_dim=UNET_INPUT_DIM, dim_mults=DIM_MULTS,
    local_dim=LOCAL_DIM, global_dim=GLOBAL_DIM,
).to(device)
```
Sanity-check: `score_net(_x, _s, _xs, _xg, torch.zeros(4, 1, MAP_H, MAP_W, device=device))`

**Cell 10 (Training Loop):** Inside loop:
```python
traj, ellipsoids = next(loader_iter)
traj       = traj.to(device)
ellipsoids = ellipsoids.to(device)
ells_norm  = normalize_ellipsoids(ellipsoids, xy_min_t.to(device), xy_range_t.to(device))
occ_map    = rasterize_scene(ells_norm, wall_bitmap, MAP_H, MAP_W)   # [B, 1, H, W]
loss, info = ve(traj, xs, xg, occ_map, p_uncond=P_UNCOND)
```

**Preview function:** Rasterize `fix_ells` before sampling, pass `occ_map` to sampler.

**Checkpoint config:** Replace ellipsoid keys (`ellipsoid_hidden_dim`, `ellipsoid_output_dim`,
`max_ellipsoids`) with `local_dim`, `global_dim`, `map_h`, `map_w`.

**Load cell:** Reconstruct with `local_dim`/`global_dim`; recompute wall bitmap from stored `xy_min`/`xy_max`.

---

## Coordinate System

- Model and trajectories operate in normalized **[-1, 1]** space (empirical `xy_min`/`xy_range`)
- Ellipsoid centers/radii from dataset are **world coords** → normalized before rasterization via `normalize_ellipsoids`
- Wall bitmap rendered by mapping [-1,1] pixel grid to world coords and testing against unit wall cells;
  uses same `xy_min`/`xy_range` so wall pixels align with trajectory space
- Outer maze walls (±5.5 world) partially exceed the empirical data range → appear at bitmap edge (correct)
