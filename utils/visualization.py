"""
Visualisation helpers for trajectory and CBF data.

Extracted from notebooks/train_and_sample_circles_copy.ipynb.
Requires matplotlib (and optionally plotly).
"""

from __future__ import annotations

import numpy as np
import torch


def plot_trajectory_snapshot(
    traj: torch.Tensor,          # [T, 2]
    obstacles: torch.Tensor,     # [N, 3] (px, py, r)
    x_start: torch.Tensor,       # [2]
    x_goal: torch.Tensor,        # [2]
    cbf_metrics: dict = None,    # from compute_cbf_metrics()
    title: str = 'Trajectory',
    ax=None,
):
    """
    Matplotlib snapshot of a single trajectory with obstacle rings.

    If cbf_metrics is provided, waypoints are coloured red (violated) or green (safe).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    pts = traj.detach().cpu().numpy()  # [T, 2]

    # --- Obstacle circles ---
    for (px, py, r) in obstacles.cpu().numpy():
        ax.add_patch(patches.Circle((px, py), r, color='red', alpha=0.15))
        ax.add_patch(patches.Circle((px, py), r, fill=False, color='red', linewidth=1.5))

    # --- Trajectory line ---
    ax.plot(pts[:, 0], pts[:, 1], '-', color='steelblue', linewidth=1.2, zorder=2)

    # --- Waypoints ---
    if cbf_metrics is not None:
        d_raw = cbf_metrics['d_raw']
        colors = ['red' if d < 0 else 'green' for d in d_raw]
        for i, (x, y) in enumerate(pts):
            ax.plot(x, y, 'o', color=colors[i], markersize=4, zorder=3)
    else:
        ax.plot(pts[:, 0], pts[:, 1], 'o', color='steelblue', markersize=3, zorder=3)

    # --- Start / Goal markers ---
    s = x_start.cpu().numpy()
    g = x_goal.cpu().numpy()
    ax.plot(s[0], s[1], 's', color='lime', markersize=10, zorder=5, label='Start')
    ax.plot(g[0], g[1], '*', color='gold', markersize=14, zorder=5, label='Goal')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_plain_vs_safe(
    plain_traj: torch.Tensor,    # [T, 2]
    safe_traj: torch.Tensor,     # [T, 2]
    prior_traj: torch.Tensor,    # [T, 2]
    obstacles: torch.Tensor,     # [N, 3]
    x_start: torch.Tensor,       # [2]
    x_goal: torch.Tensor,        # [2]
    cbf_metrics: dict = None,
    title: str = 'Plain vs Safe',
    ax=None,
):
    """
    Overlay plot of prior (grey dashed), plain (orange dashed), safe (blue solid).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    def to_np(t):
        return t.detach().cpu().numpy()

    # Obstacle
    for (px, py, r) in to_np(obstacles):
        ax.add_patch(patches.Circle((px, py), r, color='red', alpha=0.15))
        ax.add_patch(patches.Circle((px, py), r, fill=False, color='red', linewidth=1.5))

    # Prior
    p = to_np(prior_traj)
    ax.plot(p[:, 0], p[:, 1], '--', color='grey', alpha=0.3, linewidth=1.0, label='Prior (step 0)')

    # Plain
    pl = to_np(plain_traj)
    ax.plot(pl[:, 0], pl[:, 1], '--', color='orange', alpha=0.6, linewidth=1.5, label='Plain DPM')

    # Safe
    sa = to_np(safe_traj)
    ax.plot(sa[:, 0], sa[:, 1], '-', color='steelblue', linewidth=2.0, label='Safe DPM')

    if cbf_metrics is not None:
        d_raw = cbf_metrics['d_raw']
        colors = ['red' if d < 0 else 'limegreen' for d in d_raw]
        for i, (x, y) in enumerate(sa):
            ax.plot(x, y, 'o', color=colors[i], markersize=5, zorder=4)

    s = to_np(x_start)
    g = to_np(x_goal)
    ax.plot(s[0], s[1], 's', color='lime', markersize=12, zorder=6, label='Start')
    ax.plot(g[0], g[1], '*', color='gold', markersize=16, zorder=6, label='Goal')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
