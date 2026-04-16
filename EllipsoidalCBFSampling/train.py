"""
train.py — Large Maze Ellipsoids Diffuser Training Script
==========================================================
Converted from train_large_ellipsoids_diffuser.ipynb.

Live monitoring
---------------
  Terminal  : tqdm progress bar with rolling loss, sigma, step/s
  Browser   : TensorBoard — run in a separate terminal:
                tensorboard --logdir EllipsoidalCBFSampling/runs
              then open http://localhost:6006

Usage
-----
  cd SDPMSP_Clone
  conda activate py_3_10
  python -m EllipsoidalCBFSampling.train [--resume]
"""

import sys
import os
import copy
import time
import argparse
import io

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt

import minari

# ---------------------------------------------------------------------------
# Path setup: project root must be on sys.path
# ---------------------------------------------------------------------------
_NOTEBOOK_DIR  = os.path.dirname(os.path.abspath(__file__))   # EllipsoidalCBFSampling/
_PROJECT_ROOT  = os.path.dirname(_NOTEBOOK_DIR)               # SDPMSP_Clone/
for p in [_PROJECT_ROOT, _NOTEBOOK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from EllipsoidalCBFSampling.models.score_net_ellipsoids import TemporalUnet
from EllipsoidalCBFSampling.models.ve_diffusion_ellipsoids import VEDiffusion
from EllipsoidalCBFSampling.models.samplers_ellipsoids_cfg import dpm_solver_1_cfg_sample
from EllipsoidalCBFSampling.rasterize import (
    get_large_maze_wall_bitmap,
    normalize_ellipsoids,
    rasterize_scene,
)

# ===========================================================================
# Hyperparameters
# ===========================================================================

# ---- Dataset ----
DATASET_ID      = "custom/largemaze_ellipsoids-v1"
LOCAL_DATA_PATH = os.path.join(_NOTEBOOK_DIR, "../Diffuser/TrajectoryDatasetGeneration/data")
MAX_ELLIPSOIDS  = 5

# ---- Rasterized map ----
MAP_H = 64
MAP_W = 64

# ---- Model ----
UNET_INPUT_DIM = 32
DIM_MULTS      = (1, 2, 4)
LOCAL_DIM      = 16
GLOBAL_DIM     = 256

# ---- VE noise schedule ----
SIGMA_MIN = 0.01
SIGMA_MAX = 10.0
N_LEVELS  = 1000

# ---- CFG ----
P_UNCOND       = 0.1
GUIDANCE_SCALE = 0.6

# ---- Training ----
BATCH_SIZE       = 128
LR               = 3e-4
TOTAL_STEPS      = 1_000_000
EMA_DECAY        = 0.9999
EMA_START_STEP   = 5_000
EMA_UPDATE_EVERY = 10

# ---- Sampling (preview) ----
N_SAMPLE_STEPS = 25

# ---- Logging / checkpointing ----
LOG_EVERY     = 100     # tqdm postfix + tensorboard scalar update
PREVIEW_EVERY = 5_000     # write trajectory image grid to tensorboard
SAVE_EVERY    = 10_000

CKPT_DIR  = os.path.join(_NOTEBOOK_DIR, "checkpoints", "rasterized")

RUNS_DIR  = os.path.join(_NOTEBOOK_DIR, "runs")


# ===========================================================================
# Dataset
# ===========================================================================

class EllipsoidsMinariDataset(Dataset):
    """
    Wraps the custom/largemaze_ellipsoids-v1 Minari dataset.

    Returns per item:
        traj       : [T_steps, 2]        float32  — x,y waypoints, normalised to [-1, 1]
        ellipsoids : [MAX_ELLIPSOIDS, 4] float32  — (cx, cy, a, b); zeros for absent ones
    """

    def __init__(self, dataset_id, local_data_path, max_ellipsoids=5, T_steps=None):
        os.environ["MINARI_DATASETS_PATH"] = os.path.abspath(local_data_path)
        ds = minari.load_dataset(dataset_id)

        trajs      = []
        ellipsoids = []

        for ep in ds.iterate_episodes():
            obs = ep.observations
            trajs.append(obs[:, :2].astype(np.float32))

            centers = np.array(ep.infos["ellipsoids_centers"][0], dtype=np.float32)
            radii   = np.array(ep.infos["ellipsoids_radii"][0],   dtype=np.float32)
            ell     = np.concatenate([centers, radii], axis=-1)

            N = ell.shape[0]
            if N < max_ellipsoids:
                pad = np.zeros((max_ellipsoids - N, 4), dtype=np.float32)
                ell = np.concatenate([ell, pad], axis=0)
            else:
                ell = ell[:max_ellipsoids]

            ellipsoids.append(ell)

        horizon = T_steps if T_steps is not None else trajs[0].shape[0]
        self.horizon = horizon

        fixed_trajs = []
        for t in trajs:
            if t.shape[0] >= horizon:
                fixed_trajs.append(t[:horizon])
            else:
                pad_rows = np.repeat(t[-1:], horizon - t.shape[0], axis=0)
                fixed_trajs.append(np.concatenate([t, pad_rows], axis=0))

        all_traj = np.stack(fixed_trajs, axis=0)

        self.xy_min = all_traj.reshape(-1, 2).min(axis=0)
        self.xy_max = all_traj.reshape(-1, 2).max(axis=0)
        xy_range = self.xy_max - self.xy_min
        xy_range[xy_range < 1e-6] = 1.0
        self.xy_range = xy_range

        norm_traj = 2.0 * (all_traj - self.xy_min) / self.xy_range - 1.0

        self.trajs      = torch.tensor(norm_traj,            dtype=torch.float32)
        self.ellipsoids = torch.tensor(np.stack(ellipsoids), dtype=torch.float32)

        print(f"EllipsoidsMinariDataset: {len(self)} episodes, "
              f"horizon={self.horizon}, max_ellipsoids={max_ellipsoids}")
        print(f"  xy_min={self.xy_min}  xy_max={self.xy_max}")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.ellipsoids[idx]

    def unnormalize(self, x):
        xy_min   = torch.tensor(self.xy_min,   dtype=x.dtype, device=x.device)
        xy_range = torch.tensor(self.xy_range, dtype=x.dtype, device=x.device)
        return (x + 1.0) / 2.0 * xy_range + xy_min

    def normalize(self, x):
        xy_min   = torch.tensor(self.xy_min,   dtype=x.dtype, device=x.device)
        xy_range = torch.tensor(self.xy_range, dtype=x.dtype, device=x.device)
        return 2.0 * (x - xy_min) / xy_range - 1.0


# ===========================================================================
# Visualisation helpers
# ===========================================================================

_MAZE_MAP = [
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


def _draw_maze(ax):
    for i, row in enumerate(_MAZE_MAP):
        for j, cell in enumerate(row):
            if cell == 1:
                cx = -5.5 + j * 1.0
                cy =  4.0 - i * 1.0
                rect = plt.Rectangle(
                    (cx - 0.5, cy - 0.5), 1.0, 1.0,
                    linewidth=0, facecolor="dimgray",
                )
                ax.add_patch(rect)
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-4.5, 4.5)


def _draw_ellipsoid(ax, cx, cy, a, b, color="steelblue", alpha=0.4):
    t = np.linspace(0, 2 * np.pi, 200)
    x = cx + a * np.sign(np.cos(t)) * np.sqrt(np.abs(np.cos(t)))
    y = cy + b * np.sign(np.sin(t)) * np.sqrt(np.abs(np.sin(t)))
    polygon = plt.Polygon(
        np.column_stack([x, y]),
        facecolor=color, edgecolor=color, alpha=alpha,
    )
    ax.add_patch(polygon)


def _fig_to_tensor(fig):
    """Convert a matplotlib figure to a [3, H, W] uint8 tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    import PIL.Image
    img = PIL.Image.open(buf).convert("RGB")
    arr = np.array(img)          # [H, W, 3]
    return torch.tensor(arr).permute(2, 0, 1)   # [3, H, W]


def build_preview_figure(
    ema_model, ve, ds,
    fix_xs, fix_xg, fix_occ, fix_ells,
    step, device,
):
    """Sample 4 trajectories and return a matplotlib figure (not displayed inline)."""
    ema_model.eval()
    n = fix_xs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        f"Step {step:,}  —  DPM-Solver-1 CFG  "
        f"(w={GUIDANCE_SCALE}, {N_SAMPLE_STEPS} steps)",
        fontsize=11,
    )

    with torch.no_grad():
        for j, ax in enumerate(axes):
            xs_j  = fix_xs[j:j+1]
            xg_j  = fix_xg[j:j+1]
            occ_j = fix_occ[j:j+1]

            samp = dpm_solver_1_cfg_sample(
                ema_model, ve,
                x_start=xs_j, x_goal=xg_j,
                occ_map=occ_j,
                T_steps=ds.horizon,
                n_steps=N_SAMPLE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                device=device,
            )   # [1, T, 2]

            world = ds.unnormalize(samp.cpu())
            xs_w  = ds.unnormalize(xs_j.cpu())
            xg_w  = ds.unnormalize(xg_j.cpu())

            _draw_maze(ax)

            ell_np = fix_ells[j].numpy()
            for row in ell_np:
                cx, cy, a, b = row
                if a > 1e-4 or b > 1e-4:
                    _draw_ellipsoid(ax, cx, cy, a, b)

            ax.plot(world[0, :, 0], world[0, :, 1], lw=1.2, color="steelblue", alpha=0.85)
            ax.scatter(xs_w[0, 0], xs_w[0, 1], c="green", s=60, zorder=5)
            ax.scatter(xg_w[0, 0], xg_w[0, 1], c="red",   s=60, zorder=5)
            ax.set_title(f"pair {j}")
            ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LargeMaze Ellipsoids Diffuser")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint at CKPT_PATH")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help=f"Total training steps (default: {TOTAL_STEPS})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Dataset + DataLoader
    # -----------------------------------------------------------------------
    ds = EllipsoidsMinariDataset(
        DATASET_ID, LOCAL_DATA_PATH, max_ellipsoids=MAX_ELLIPSOIDS
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=(device == "cuda"))
    T_STEPS = ds.horizon
    print(f"T_STEPS={T_STEPS}, batches/epoch={len(dl)}")

    xy_min_t   = torch.tensor(ds.xy_min,   dtype=torch.float32).to(device)
    xy_range_t = torch.tensor(ds.xy_range, dtype=torch.float32).to(device)
    wall_bitmap = get_large_maze_wall_bitmap(ds.xy_min, ds.xy_range, MAP_H, MAP_W).to(device)
    print(f"wall_bitmap: {tuple(wall_bitmap.shape)}, "
          f"wall px = {wall_bitmap.sum().int().item()} / {MAP_H * MAP_W}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    score_net = TemporalUnet(
        state_dim=2,
        T_steps=T_STEPS,
        unet_input_dim=UNET_INPUT_DIM,
        dim_mults=DIM_MULTS,
        local_dim=LOCAL_DIM,
        global_dim=GLOBAL_DIM,
    ).to(device)

    ve = VEDiffusion(
        model=score_net,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        n_levels=N_LEVELS,
    ).to(device)

    ema_model = copy.deepcopy(score_net).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(score_net.parameters(), lr=LR)

    start_step   = 0
    loss_history = []

    # -----------------------------------------------------------------------
    # Optional resume
    # -----------------------------------------------------------------------
    # Find the latest checkpoint by step number
    _resume_path = None
    if args.resume and os.path.isdir(CKPT_DIR):
        ckpts = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")],
            key=lambda f: int(f.split("_step")[-1].replace(".pt", "") or 0),
        )
        if ckpts:
            _resume_path = os.path.join(CKPT_DIR, ckpts[-1])

    if _resume_path:
        print(f"Resuming from {_resume_path} ...")
        ckpt = torch.load(_resume_path, map_location=device, weights_only=False)
        score_net.load_state_dict(ckpt["score_net"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        loss_history = ckpt.get("loss_history", [])
        start_step   = ckpt["step"]
        print(f"  resumed at step {start_step:,}")
    else:
        n_params = sum(p.numel() for p in score_net.parameters())
        print(f"TemporalUnet params: {n_params:,}")

    # -----------------------------------------------------------------------
    # Fixed preview samples (4 random episodes, seeded for reproducibility)
    # -----------------------------------------------------------------------
    rng_fix  = np.random.default_rng(42)
    fix_idx  = rng_fix.choice(len(ds), size=4, replace=False).tolist()
    fix_trajs, fix_ells = zip(*[ds[i] for i in fix_idx])
    fix_trajs = torch.stack(fix_trajs)   # [4, T, 2]
    fix_ells  = torch.stack(fix_ells)    # [4, 5, 4]
    fix_xs    = fix_trajs[:, 0, :].to(device)
    fix_xg    = fix_trajs[:, -1, :].to(device)

    fix_ells_norm = normalize_ellipsoids(
        fix_ells.to(device), xy_min_t, xy_range_t
    )
    fix_occ = rasterize_scene(fix_ells_norm, wall_bitmap, MAP_H, MAP_W)   # [4, 1, H, W]

    # -----------------------------------------------------------------------
    # TensorBoard writer
    # -----------------------------------------------------------------------
    writer = SummaryWriter(log_dir=RUNS_DIR)
    print(f"\nTensorBoard logs → {RUNS_DIR}")
    print("  Run in another terminal:  tensorboard --logdir EllipsoidalCBFSampling/runs")
    print("  Then open:               http://localhost:6006\n")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    import itertools
    loader_iter = itertools.cycle(dl)

    total_steps   = args.steps
    remaining     = total_steps - start_step
    recent_losses = []

    score_net.train()
    pbar = tqdm(
        range(start_step + 1, total_steps + 1),
        initial=start_step,
        total=total_steps,
        dynamic_ncols=True,
        desc="training",
    )

    t0 = time.time()

    for step in pbar:
        traj, ellipsoids = next(loader_iter)
        traj       = traj.to(device)
        ellipsoids = ellipsoids.to(device)

        xs = traj[:, 0, :]
        xg = traj[:, -1, :]

        ells_norm = normalize_ellipsoids(ellipsoids, xy_min_t, xy_range_t)
        occ_map   = rasterize_scene(ells_norm, wall_bitmap, MAP_H, MAP_W)

        loss, info = ve(traj, xs, xg, occ_map, p_uncond=P_UNCOND)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        if step >= EMA_START_STEP and step % EMA_UPDATE_EVERY == 0:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), score_net.parameters()):
                    p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

        loss_val = info["loss"]
        loss_history.append(loss_val)
        recent_losses.append(loss_val)

        # -- tqdm + TensorBoard scalars --
        if step % LOG_EVERY == 0:
            avg_loss   = float(np.mean(recent_losses))
            sigma_mean = float(info["sigma_mean"])
            recent_losses.clear()

            writer.add_scalar("train/loss",       avg_loss,   step)
            writer.add_scalar("train/sigma_mean", sigma_mean, step)

            elapsed = time.time() - t0
            steps_done = step - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0.0

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                sigma=f"{sigma_mean:.3f}",
                sps=f"{sps:.1f}",
            )

        # -- TensorBoard trajectory preview --
        if step % PREVIEW_EVERY == 0:
            fig = build_preview_figure(
                ema_model, ve, ds,
                fix_xs, fix_xg, fix_occ, fix_ells,
                step, device,
            )
            img_tensor = _fig_to_tensor(fig)   # [3, H, W]
            writer.add_image("samples/trajectories", img_tensor, step)
            plt.close(fig)
            score_net.train()

        # -- Checkpoint --
        if step % SAVE_EVERY == 0:
            _save_checkpoint(
                step, score_net, ema_model, optimizer,
                loss_history, T_STEPS, ds,
            )
            writer.flush()

    # Final save
    _save_checkpoint(step, score_net, ema_model, optimizer, loss_history, T_STEPS, ds)
    writer.close()
    pbar.close()
    print("Training complete.")


def _save_checkpoint(step, score_net, ema_model, optimizer, loss_history, T_STEPS, ds):
    torch.save(
        {
            "step":      step,
            "score_net": score_net.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": {
                "T_steps":        T_STEPS,
                "unet_input_dim": UNET_INPUT_DIM,
                "dim_mults":      list(DIM_MULTS),
                "sigma_min":      SIGMA_MIN,
                "sigma_max":      SIGMA_MAX,
                "n_levels":       N_LEVELS,
                "local_dim":      LOCAL_DIM,
                "global_dim":     GLOBAL_DIM,
                "map_h":          MAP_H,
                "map_w":          MAP_W,
                "p_uncond":       P_UNCOND,
                "guidance_scale": GUIDANCE_SCALE,
                "dataset_id":     DATASET_ID,
                "xy_min":         ds.xy_min.tolist(),
                "xy_max":         ds.xy_max.tolist(),
            },
        },
        os.path.join(CKPT_DIR, f"ve_unet_largemaze_rasterized_step{step}.pt"),
    )
    print(f"  [step {step:,}] checkpoint saved → {CKPT_DIR}/ve_unet_largemaze_rasterized_step{step}.pt")


if __name__ == "__main__":
    main()
