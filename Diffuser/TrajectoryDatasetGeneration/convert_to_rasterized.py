"""
convert_to_rasterized.py

Converts any Minari-format HDF5 dataset (with ellipsoid obstacles) into a
rasterized numpy dataset suitable for training the rasterized local-feat model.

Usage:
    python convert_to_rasterized.py <dataset_root> [options]

    <dataset_root>   Path to the dataset root directory that contains
                     data/main_data.hdf5  (e.g. data/custom/largemaze_ellipsoids_2-v1)

Options:
    --out  DIR       Output directory. Default: <dataset_root>_rasterized
    --H    INT       Bitmap height in pixels. Default: 64
    --W    INT       Bitmap width  in pixels. Default: 64
    --walls          Overlay the large-maze wall layout (largemaze datasets only)

Output layout (<out>/):
    metadata.json          — dataset-wide normalization constants and schema
    episodes/
        episode_0.npz
        episode_1.npz
        ...

Each episode_N.npz contains:
    occ_map          [1, H, W]  binary occupancy (walls + ellipsoids), float32
    trajectory       [T, 4]     (x, y, vx, vy) world coords,           float32
    trajectory_norm  [T, 4]     (x_n, y_n, vx_n, vy_n) normalized,     float32
                                xy -> [-1,1]; vxvy scaled by same factor (no offset)
    start_pos        [2]        world coords,                           float32
    goal_pos         [2]        world coords,                           float32
    start_norm       [2]        normalized [-1,1],                      float32
    goal_norm        [2]        normalized [-1,1],                      float32
    ellipsoids_world [5, 4]     (cx, cy, a, b) world coords,           float32
                                zero-padded for absent ellipsoids
    xy_min           [2]        dataset-wide empirical x_min, y_min
    xy_range         [2]        dataset-wide empirical x_range, y_range

Normalization convention (matches rasterize.py / score_net_rasterizedlocalfeats.py):
    x_norm  = 2 * (x_world  - xy_min[0]) / xy_range[0] - 1
    y_norm  = 2 * (y_world  - xy_min[1]) / xy_range[1] - 1
    vx_norm = 2 * vx_world / xy_range[0]   (scale only, no offset)
    vy_norm = 2 * vy_world / xy_range[1]
"""

import sys
import os
import json
import argparse
import numpy as np
import h5py
import torch

# ---------------------------------------------------------------------------
# Project root on path so EllipsoidalCBFSampling can be imported
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from EllipsoidalCBFSampling.rasterize import (
    get_large_maze_wall_bitmap,
    normalize_ellipsoids,
    rasterize_scene,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_xy(xy_world, xy_min, xy_range):
    """world (x,y) → normalized [-1,1]. xy: (..., 2)"""
    return 2.0 * (xy_world - xy_min) / xy_range - 1.0


def normalize_vxvy(vxvy_world, xy_range):
    """velocity → scale only (no offset). vxvy: (..., 2)"""
    return 2.0 * vxvy_world / xy_range


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a Minari-format HDF5 ellipsoid dataset to per-episode rasterized .npz files."
    )
    p.add_argument(
        'dataset_root',
        help="Path to the dataset root directory containing data/main_data.hdf5 "
             "(e.g. data/custom/largemaze_ellipsoids_2-v1)",
    )
    p.add_argument(
        '--out', default=None,
        help="Output directory. Default: <dataset_root>_rasterized",
    )
    p.add_argument('--H', type=int, default=64, help="Bitmap height in pixels (default: 64)")
    p.add_argument('--W', type=int, default=64, help="Bitmap width  in pixels (default: 64)")
    p.add_argument(
        '--walls', action='store_true',
        help="Overlay the large-maze wall layout onto the occupancy map (largemaze datasets only)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    src_hdf5     = os.path.join(dataset_root, 'data', 'main_data.hdf5')
    out_dir      = args.out if args.out else dataset_root.rstrip('/\\') + '_rasterized'
    out_dir      = os.path.abspath(out_dir)
    H, W         = args.H, args.W
    use_walls    = args.walls

    if not os.path.isfile(src_hdf5):
        raise FileNotFoundError(f"HDF5 not found: {src_hdf5}")

    dataset_name = os.path.relpath(dataset_root,
                                   os.path.join(dataset_root, '..', '..')).replace('\\', '/')
    print(f"Source : {src_hdf5}")
    print(f"Output : {out_dir}")
    print(f"Bitmap : {H}x{W},  walls={'yes' if use_walls else 'no'}")

    os.makedirs(out_dir, exist_ok=True)
    f = h5py.File(src_hdf5, 'r')

    ep_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
    N = len(ep_keys)
    print(f"Found {N} episodes")

    # --- Dataset-wide normalization constants from all observations ---
    all_xy = np.concatenate(
        [f[k]['observations'][:, :2] for k in ep_keys], axis=0
    )
    xy_min   = all_xy.min(axis=0).astype(np.float32)
    xy_max   = all_xy.max(axis=0).astype(np.float32)
    xy_range = (xy_max - xy_min).astype(np.float32)
    print(f"xy_min={xy_min}, xy_max={xy_max}, xy_range={xy_range}")

    xy_min_t   = torch.tensor(xy_min)
    xy_range_t = torch.tensor(xy_range)

    # --- Pre-render wall bitmap (shared, zeros if --no-walls) ---
    if use_walls:
        wall_bitmap = get_large_maze_wall_bitmap(xy_min_t, xy_range_t, H, W)
        print("Wall bitmap rendered.")
    else:
        wall_bitmap = torch.zeros(1, 1, H, W)
        print("Wall bitmap skipped (--no-walls).")

    # --- Per-episode output directory ---
    ep_dir = os.path.join(out_dir, 'episodes')
    os.makedirs(ep_dir, exist_ok=True)

    T = f[ep_keys[0]]['observations'].shape[0]

    for idx, ep_key in enumerate(ep_keys):
        ep = f[ep_key]

        # -- Trajectory (x, y, vx, vy) --
        obs      = ep['observations'][:].astype(np.float32)          # [T, 4]
        xy_obs   = obs[:, :2]
        vxvy_obs = obs[:, 2:]
        xy_norm   = normalize_xy(xy_obs, xy_min, xy_range)
        vxvy_norm = normalize_vxvy(vxvy_obs, xy_range)
        traj_norm = np.concatenate([xy_norm, vxvy_norm], axis=1)     # [T, 4]

        # -- Start / goal (constant per episode, take step 0) --
        start_w = ep['infos/start_pos'][0].astype(np.float32)        # [2]
        goal_w  = ep['infos/goal_pos'][0].astype(np.float32)         # [2]

        # -- Ellipsoids (constant per episode, take step 0) --
        centers   = ep['infos/ellipsoids_centers'][0].astype(np.float32)  # [N_ell, 2]
        radii     = ep['infos/ellipsoids_radii'][0].astype(np.float32)    # [N_ell, 2]
        ell_world = np.concatenate([centers, radii], axis=1)              # [N_ell, 4]

        # -- Rasterize: walls + ellipsoids --
        ell_t    = torch.tensor(ell_world[None])                          # [1, N_ell, 4]
        ell_norm = normalize_ellipsoids(ell_t, xy_min_t, xy_range_t)     # [1, N_ell, 4]
        occ      = rasterize_scene(ell_norm, wall_bitmap, H, W)          # [1, 1, H, W]
        occ_map  = occ[0].numpy()                                         # [1, H, W]

        # -- Save episode file --
        ep_path = os.path.join(ep_dir, f'episode_{idx}.npz')
        np.savez_compressed(
            ep_path,
            occ_map          = occ_map,
            trajectory       = obs,
            trajectory_norm  = traj_norm,
            start_pos        = start_w,
            goal_pos         = goal_w,
            start_norm       = normalize_xy(start_w, xy_min, xy_range),
            goal_norm        = normalize_xy(goal_w,  xy_min, xy_range),
            ellipsoids_world = ell_world,
            xy_min           = xy_min,
            xy_range         = xy_range,
        )

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{N} episodes")

    f.close()

    meta = {
        "source_dataset": dataset_name,
        "n_episodes": N,
        "T_steps": T,
        "bitmap_H": H,
        "bitmap_W": W,
        "walls_included": use_walls,
        "xy_min": xy_min.tolist(),
        "xy_range": xy_range.tolist(),
        "episode_files": "episodes/episode_N.npz  (N = 0 .. n_episodes-1)",
        "arrays_per_episode": {
            "occ_map":          f"[1, {H}, {W}] float32 - binary occupancy (walls+ellipsoids)",
            "trajectory":        "[T, 4] float32 - (x,y,vx,vy) world coords",
            "trajectory_norm":   "[T, 4] float32 - (x_n,y_n,vx_n,vy_n) normalized",
            "start_pos":         "[2] float32 - world coords",
            "goal_pos":          "[2] float32 - world coords",
            "start_norm":        "[2] float32 - normalized [-1,1]",
            "goal_norm":         "[2] float32 - normalized [-1,1]",
            "ellipsoids_world":  "[N_ell, 4] float32 - (cx,cy,a,b) world coords; zero-padded if absent",
            "xy_min":            "[2] float32 - dataset-wide normalization min",
            "xy_range":          "[2] float32 - dataset-wide normalization range",
        },
        "normalization": {
            "xy":   "2*(world - xy_min)/xy_range - 1",
            "vxvy": "2*vel/xy_range  (scale only, no offset)",
        },
    }
    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)
    print(f"Saved metadata: {meta_path}")

    # --- Quick sanity check on episode 0 ---
    data = np.load(os.path.join(ep_dir, 'episode_0.npz'))
    print("\n=== Sanity check (episode_0) ===")
    for k, v in data.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")


if __name__ == '__main__':
    main()
