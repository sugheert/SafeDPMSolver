"""
rasterize.py — Scene rasterization utilities for the rasterized environment encoder.

Provides three functions:

    get_large_maze_wall_bitmap(xy_min, xy_range, H, W)
        Pre-render the fixed large-maze wall layout as a binary occupancy bitmap
        in the model's normalized [-1, 1] coordinate space.  Call once at training
        start and cache the result on GPU.

    normalize_ellipsoids(ellipsoids_world, xy_min, xy_range)
        Convert ellipsoid parameters from world coordinates (as stored in the
        Minari dataset) to the model's normalized [-1, 1] space, so they can be
        rasterized into the same bitmap as the walls.

    rasterize_scene(ellipsoids_norm, wall_bitmap, H, W)
        Rasterize a batch of normalized ellipsoids and merge with the pre-rendered
        wall bitmap, producing the [B, 1, H, W] occupancy map fed to MapEncoder.

Coordinate conventions
----------------------
World frame  : X ∈ [-6, 6],  Y ∈ [-4.5, 4.5]  (large maze extents)
Normalised   : [-1, 1] mapped via empirical xy_min / xy_range from the dataset
Bitmap frame : row 0 = y = -1 (bottom), row H-1 = y = +1 (top)
               col 0 = x = -1 (left),   col W-1 = x = +1 (right)
               (matches torch.meshgrid with indexing='ij' and linspace(-1, 1, H/W))
"""

import torch


# ---------------------------------------------------------------------------
# Large-maze layout  (matches _MAZE_MAP in train_large_ellipsoids_diffuser.ipynb)
# ---------------------------------------------------------------------------

_LARGE_MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Wall cell world-coordinate formula (matches draw_maze in the notebook):
#   cx = -5.5 + j * 1.0   (column index j)
#   cy =  4.0 - i * 1.0   (row    index i)
# Each cell is a unit square: [cx - 0.5, cx + 0.5] × [cy - 0.5, cy + 0.5]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_large_maze_wall_bitmap(
    xy_min,         # array-like [2]  empirical dataset xy_min  (world coords)
    xy_range,       # array-like [2]  empirical dataset xy_range (world coords)
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:
    """
    Pre-render the large-maze wall layout as a binary occupancy bitmap.

    Args:
        xy_min   : [x_min, y_min] in world coordinates (dataset empirical min).
        xy_range : [x_range, y_range] in world coordinates (xy_max - xy_min).
        H, W     : bitmap height and width in pixels (default 64 × 64).

    Returns:
        Tensor of shape [1, 1, H, W], dtype float32, values in {0, 1}.
        Broadcast-ready over a batch dimension.
    """
    # Build a pixel grid in normalised [-1, 1] space
    ys = torch.linspace(-1.0, 1.0, H)
    xs = torch.linspace(-1.0, 1.0, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W] each

    # Convert normalised coords to world coords using the dataset normalisation
    # norm = 2 * (world - xy_min) / xy_range - 1
    # → world = (norm + 1) / 2 * xy_range + xy_min
    xmin_x, xmin_y  = float(xy_min[0]),   float(xy_min[1])
    xrng_x, xrng_y  = float(xy_range[0]), float(xy_range[1])

    gx_world = (grid_x + 1.0) / 2.0 * xrng_x + xmin_x   # [H, W]
    gy_world = (grid_y + 1.0) / 2.0 * xrng_y + xmin_y   # [H, W]

    bitmap = torch.zeros(1, 1, H, W, dtype=torch.float32)

    for i, row in enumerate(_LARGE_MAZE_MAP):
        for j, cell in enumerate(row):
            if cell == 1:
                cx = -5.5 + j * 1.0   # wall cell centre x (world)
                cy =  4.0 - i * 1.0   # wall cell centre y (world)
                inside = (
                    (gx_world - cx).abs() <= 0.5
                ) & (
                    (gy_world - cy).abs() <= 0.5
                )
                bitmap[0, 0] = (bitmap[0, 0] + inside.float()).clamp(0.0, 1.0)

    return bitmap   # [1, 1, H, W]


def normalize_ellipsoids(
    ellipsoids_world: torch.Tensor,   # [B, N, 4]  (cx, cy, a, b) in world coords
    xy_min:           torch.Tensor,   # [2]         dataset empirical xy_min
    xy_range:         torch.Tensor,   # [2]         dataset empirical xy_range
) -> torch.Tensor:
    """
    Map ellipsoid parameters from world coordinates to the model's normalised
    [-1, 1] space so they can be rasterized together with the wall bitmap.

    Mapping:
        cx_norm = 2 * (cx_world - xy_min[0]) / xy_range[0] - 1
        cy_norm = 2 * (cy_world - xy_min[1]) / xy_range[1] - 1
        a_norm  = 2 * a_world  / xy_range[0]   (scale only — no offset)
        b_norm  = 2 * b_world  / xy_range[1]

    Zero-padded ellipsoids (a == 0) are preserved as zeros; the `a > 0` guard
    in rasterize_scene skips them automatically.

    Args:
        ellipsoids_world : [B, N, 4] tensor of (cx, cy, a, b) in world coords.
        xy_min           : [2] tensor, same device as ellipsoids_world.
        xy_range         : [2] tensor, same device as ellipsoids_world.

    Returns:
        [B, N, 4] tensor in normalised [-1, 1] space.
    """
    out = ellipsoids_world.clone()
    out[..., 0] = 2.0 * (ellipsoids_world[..., 0] - xy_min[0]) / xy_range[0] - 1.0
    out[..., 1] = 2.0 * (ellipsoids_world[..., 1] - xy_min[1]) / xy_range[1] - 1.0
    out[..., 2] = 2.0 * ellipsoids_world[..., 2] / xy_range[0]
    out[..., 3] = 2.0 * ellipsoids_world[..., 3] / xy_range[1]
    return out


def rasterize_scene(
    ellipsoids_norm: torch.Tensor,   # [B, N, 4]  (cx, cy, a, b) in [-1, 1] space
    wall_bitmap:     torch.Tensor,   # [1, 1, H, W] or [B, 1, H, W] pre-rendered walls
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:
    """
    Rasterize ellipsoid obstacles onto a pixel grid and merge with the wall bitmap.

    Both walls and ellipsoids are combined into a single binary occupancy map
    that is fed to MapEncoder. Absent (zero-padded) ellipsoids are skipped via
    the `a > 0` guard.

    Args:
        ellipsoids_norm : Ellipsoid parameters in the normalised [-1, 1] space.
                          Pad absent ellipsoids with zeros.
        wall_bitmap     : Pre-rendered binary wall bitmap.  Shape [1, 1, H, W]
                          (broadcast over batch) or [B, 1, H, W].
        H, W            : Output resolution — must match wall_bitmap spatial size.

    Returns:
        Binary occupancy map [B, 1, H, W], values in {0, 1}.
    """
    B      = ellipsoids_norm.shape[0]
    device = ellipsoids_norm.device

    # Build normalised pixel grid shared across all batch items
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]
    grid_x = grid_x[None, None]   # [1, 1, H, W]
    grid_y = grid_y[None, None]

    occ = torch.zeros(B, 1, H, W, device=device)

    for i in range(ellipsoids_norm.shape[1]):
        cx = ellipsoids_norm[:, i, 0].view(B, 1, 1, 1)
        cy = ellipsoids_norm[:, i, 1].view(B, 1, 1, 1)
        a  = ellipsoids_norm[:, i, 2].view(B, 1, 1, 1)
        b  = ellipsoids_norm[:, i, 3].view(B, 1, 1, 1)

        # Skip zero-padded ellipsoids (a == 0 → absent)
        valid = (a > 0).float()

        inside = (
            ((grid_x - cx) / (a + 1e-6)) ** 2
          + ((grid_y - cy) / (b + 1e-6)) ** 2
        ) <= 1.0

        occ = occ + valid * inside.float()

    # Merge with wall bitmap (broadcast over batch if wall_bitmap is [1,1,H,W])
    wall = wall_bitmap.to(device)
    return (occ + wall).clamp(0.0, 1.0)   # [B, 1, H, W]
