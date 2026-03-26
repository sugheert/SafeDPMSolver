"""
Dataset utilities for MPD trajectory data.

Extracted from notebooks/train_and_sample_circles_copy.ipynb.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class MPDTrajectoryDataset(Dataset):
    """
    Loads MPD pre-generated trajectories as complete sequences.

    Each trajs-free.pt file contains [n_trajs, 64, 4] trajectories (x, y, vx, vy).
    We use only (x, y) and return one full trajectory [T_steps, 2] per sample,
    along with its start [2] and goal [2].

    Args:
        env_dir      : path to the environment directory containing context folders
        max_contexts : optional cap on number of context sub-directories to load
    """

    def __init__(self, env_dir: str, max_contexts: int = None):
        contexts = sorted(
            [d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))],
            key=int,
        )
        if max_contexts is not None:
            contexts = contexts[:max_contexts]

        trajs_l, starts_l, goals_l = [], [], []

        for c in contexts:
            pt_file = os.path.join(env_dir, c, 'trajs-free.pt')
            if not os.path.exists(pt_file):
                continue
            trajs = torch.load(pt_file, map_location='cpu')[..., :2].float()  # [n, 64, 2]
            for traj in trajs:
                trajs_l.append(traj)       # [64, 2]
                starts_l.append(traj[0])   # [2]
                goals_l.append(traj[-1])   # [2]

        self.trajs  = torch.stack(trajs_l)   # [N, 64, 2]
        self.starts = torch.stack(starts_l)  # [N, 2]
        self.goals  = torch.stack(goals_l)   # [N, 2]

        print(f'[MPDTrajectoryDataset]  env={os.path.basename(env_dir)}')
        print(f'  {len(self.trajs):,} trajectories | shape: {tuple(self.trajs[0].shape)}')

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.starts[idx], self.goals[idx]
