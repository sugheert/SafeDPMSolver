"""
MazeDataset — loader for the preprocessed PointMaze UMaze-v2 dataset.

Saved layout (one folder per env):
    data/umaze_v2/
        trajectories.pt    float32 [N, H, 4]  normalised  (x, y, vx, vy)
        metadata.json      horizon, obs_dim, norm bounds, maze layout, …

Usage
-----
    from maze_dataset import MazeDataset, render_maze_ax

    ds  = MazeDataset("data/umaze_v2")
    dl  = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    batch = next(iter(dl))          # [64, H, 4]  normalised
    world = ds.unnormalize(batch)   # [64, H, 4]  world coords

    fig, ax = plt.subplots()
    render_maze_ax(ax, ds.maze_map, ds.cell_size)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Maze rendering helper ─────────────────────────────────────────────────────

def render_maze_ax(ax, maze_map, cell_size=1.0, wall_color="#333333"):
    """Draw maze walls as filled rectangles on a matplotlib Axes."""
    import matplotlib.patches as patches

    n_rows = len(maze_map)
    n_cols = len(maze_map[0])

    def cell_to_xy(r, c):
        x = (c - (n_cols - 1) / 2) * cell_size
        y = ((n_rows - 1) / 2 - r) * cell_size
        return x, y

    for r in range(n_rows):
        for c in range(n_cols):
            if maze_map[r][c] == 1:
                cx, cy = cell_to_xy(r, c)
                ax.add_patch(patches.Rectangle(
                    (cx - cell_size / 2, cy - cell_size / 2),
                    cell_size, cell_size,
                    linewidth=0, facecolor=wall_color,
                ))

    half_w = n_cols / 2 * cell_size
    half_h = n_rows / 2 * cell_size
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    ax.set_aspect("equal")


# ── Dataset ───────────────────────────────────────────────────────────────────

class MazeDataset(Dataset):
    """
    Loads the preprocessed maze trajectory dataset from *data_dir*.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``trajectories.pt`` and ``metadata.json``.
    xy_only : bool
        If True, return only the (x, y) channels → shape [H, 2].
        If False, return all channels (x, y, vx, vy) → shape [H, 4].
    """

    def __init__(self, data_dir: str | Path, xy_only: bool = False):
        data_dir = Path(data_dir)

        # ── load tensor ──────────────────────────────────────────────────────
        self.trajectories = torch.load(
            data_dir / "trajectories.pt", weights_only=True
        )                                           # [N, H, obs_dim]

        # ── load metadata ────────────────────────────────────────────────────
        with open(data_dir / "metadata.json") as f:
            meta = json.load(f)

        self.horizon    = meta["horizon"]
        self.obs_dim    = meta["obs_dim"]
        self.env_id     = meta["env_id"]
        self.goal_thresh = meta["goal_thresh"]
        self.maze_map   = meta["maze_map"]          # list[list[int]]
        self.cell_size  = meta["cell_size"]

        d_min = np.array(meta["norm"]["d_min"], dtype=np.float32)
        d_max = np.array(meta["norm"]["d_max"], dtype=np.float32)

        self.xy_only = xy_only

        # When xy_only, slice norm bounds to the first 2 dims so that
        # normalize/unnormalize work on [... , 2] tensors without shape errors.
        if xy_only:
            d_min = d_min[:2]
            d_max = d_max[:2]

        self.register_buffer_norm(d_min, d_max)

    # ── norm helpers ─────────────────────────────────────────────────────────

    def register_buffer_norm(self, d_min: np.ndarray, d_max: np.ndarray):
        self._d_min  = torch.tensor(d_min)   # [obs_dim]
        self._d_max  = torch.tensor(d_max)   # [obs_dim]
        self._d_range = self._d_max - self._d_min + 1e-8

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map world-coord tensor [..., obs_dim] → [-1, 1]."""
        return 2.0 * (x - self._d_min) / self._d_range - 1.0

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised tensor [..., obs_dim] → world coords."""
        return (x + 1.0) / 2.0 * self._d_range + self._d_min

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj = self.trajectories[idx]   # [H, obs_dim]  normalised
        if self.xy_only:
            traj = traj[:, :2]
        return traj

    # ── convenience ───────────────────────────────────────────────────────────

    def sample(self, n: int, xy_only: bool | None = None) -> torch.Tensor:
        """Return *n* random trajectories (normalised)."""
        xy_only = self.xy_only if xy_only is None else xy_only
        idx = torch.randperm(len(self))[:n]
        traj = self.trajectories[idx]
        return traj[:, :, :2] if xy_only else traj

    def __repr__(self) -> str:
        return (
            f"MazeDataset(env={self.env_id!r}, "
            f"n={len(self)}, horizon={self.horizon}, "
            f"obs_dim={self.obs_dim}, xy_only={self.xy_only})"
        )


# ── Save utility (called from the preprocessing notebook) ────────────────────

def save_dataset(
    trajectories: torch.Tensor,
    d_min: np.ndarray,
    d_max: np.ndarray,
    *,
    data_dir: str | Path,
    env_id: str,
    horizon: int,
    goal_thresh: float,
    maze_map: list,
    cell_size: float = 1.0,
):
    """
    Persist a preprocessed trajectory dataset to *data_dir*.

    Parameters
    ----------
    trajectories : torch.Tensor  [N, H, obs_dim]  normalised float32
    d_min, d_max : np.ndarray    [obs_dim]  normalisation bounds (world coords)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # tensor
    torch.save(trajectories, data_dir / "trajectories.pt")

    # metadata
    meta = {
        "env_id":       env_id,
        "horizon":      horizon,
        "obs_dim":      trajectories.shape[-1],
        "n_windows":    len(trajectories),
        "goal_thresh":  goal_thresh,
        "maze_map":     maze_map,
        "cell_size":    cell_size,
        "norm": {
            "d_min": d_min.tolist(),
            "d_max": d_max.tolist(),
        },
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(trajectories)} windows → {data_dir}/")
    print(f"  trajectories.pt : {tuple(trajectories.shape)}  float32")
    print(f"  metadata.json   : horizon={horizon}, obs_dim={trajectories.shape[-1]}, "
          f"norm_min={d_min.round(3)}, norm_max={d_max.round(3)}")
