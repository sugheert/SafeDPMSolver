"""
train_rasterizedlocalfeats.py — Training script for the local-feature-sampling model.
======================================================================================

Uses pre-rasterized datasets (produced by convert_to_rasterized.py) instead of
loading from Minari HDF5 and rasterizing on-the-fly. Occupancy maps are loaded
directly from episode .npz files, eliminating per-batch rasterization overhead.

Architecture:
    - MapEncoder spatial features sampled at each waypoint (bilinear)
    - Global FiLM carries only sigma + start/goal; map conditioning is purely local.
    - U-Net input channels: state_dim + local_dim = 18

Datasets (must be pre-converted with convert_to_rasterized.py):
    Training : largemaze_ellipsoids-v1_rasterized
    Eval     : largemaze_ellipsoids_2-v1_rasterized

Usage
-----
  cd SDPMSP_Rasterized_LocalFeats
  conda activate py_3_10
  python -m EllipsoidalCBFSampling.train_rasterizedlocalfeats [--resume] [--steps N]
"""

import sys
import os
import json
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_NOTEBOOK_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_NOTEBOOK_DIR)
for p in [_PROJECT_ROOT, _NOTEBOOK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from EllipsoidalCBFSampling.models.score_net_ellipsoids_rasterizedlocalfeats import TemporalUnet
from EllipsoidalCBFSampling.models.ve_diffusion_ellipsoids_rasterizedlocalfeats import VEDiffusion
from EllipsoidalCBFSampling.models.samplers_ellipsoids_cfg_rasterizedlocalfeats import dpm_solver_1_cfg_sample

# ===========================================================================
# Hyperparameters
# ===========================================================================

_DATA_ROOT = os.path.join(_NOTEBOOK_DIR, "../Diffuser/TrajectoryDatasetGeneration/data/custom")

TRAIN_RASTERIZED_DIR = os.path.join(_DATA_ROOT, "largemaze_ellipsoids-v1_rasterized")
EVAL_RASTERIZED_DIR  = os.path.join(_DATA_ROOT, "largemaze_ellipsoids_2-v1_rasterized")

# ---- Model ----
UNET_INPUT_DIM = 32
DIM_MULTS      = (1, 2, 4)
LOCAL_DIM      = 16

# ---- VE noise schedule ----
SIGMA_MIN = 0.01
SIGMA_MAX = 10.0
N_LEVELS  = 1000

# ---- CFG ----
P_UNCOND       = 0.1
GUIDANCE_SCALE = 1.0

# ---- Training ----
BATCH_SIZE       = 64
LR               = 3e-4
TOTAL_STEPS      = 500_000
EMA_DECAY        = 0.995
EMA_START_STEP   = 5_000
EMA_UPDATE_EVERY = 10

# ---- Sampling (preview) ----
N_SAMPLE_STEPS = 25

# ---- Logging / checkpointing ----
LOG_EVERY     = 100
PREVIEW_EVERY = 5_000
SAVE_EVERY    = 5_000

CKPT_DIR = os.path.join(_NOTEBOOK_DIR, "checkpoints", "rasterized_localfeats")
RUNS_DIR = os.path.join(_NOTEBOOK_DIR, "runs")


# ===========================================================================
# Dataset
# ===========================================================================

class RasterizedEllipsoidsDataset(Dataset):
    """
    Loads a pre-rasterized dataset produced by convert_to_rasterized.py.

    Returns per item:
        traj    : [T_steps, 2]   float32 — x,y waypoints, normalised to [-1,1]
        occ_map : [1, H, W]      float32 — binary occupancy (walls + ellipsoids)

    ellipsoids_world [N, n_ell, 4] is kept as a numpy array for visualisation only.
    """

    def __init__(self, rasterized_dir, T_steps=None):
        rasterized_dir = os.path.abspath(rasterized_dir)
        with open(os.path.join(rasterized_dir, "metadata.json")) as f:
            meta = json.load(f)

        self.xy_min   = np.array(meta["xy_min"],   dtype=np.float32)
        self.xy_range = np.array(meta["xy_range"], dtype=np.float32)
        self.xy_max   = self.xy_min + self.xy_range
        ep_dir        = os.path.join(rasterized_dir, "episodes")

        trajs, occ_maps, ell_world = [], [], []

        for i in range(meta["n_episodes"]):
            ep   = np.load(os.path.join(ep_dir, f"episode_{i}.npz"))
            traj = ep["trajectory_norm"][:, :2].astype(np.float32)   # [T, 2] x,y only

            if T_steps is not None:
                if traj.shape[0] >= T_steps:
                    traj = traj[:T_steps]
                else:
                    pad  = np.repeat(traj[-1:], T_steps - traj.shape[0], axis=0)
                    traj = np.concatenate([traj, pad], axis=0)

            trajs.append(traj)
            occ_maps.append(ep["occ_map"].astype(np.float32))             # [1, H, W]
            ell_world.append(ep["ellipsoids_world"].astype(np.float32))   # [n_ell, 4]

        self.horizon          = T_steps if T_steps is not None else trajs[0].shape[0]
        self.trajs            = torch.tensor(np.stack(trajs),    dtype=torch.float32)
        self.occ_maps         = torch.tensor(np.stack(occ_maps), dtype=torch.float32)
        self.ellipsoids_world = np.stack(ell_world)   # numpy — for viz only

        print(f"RasterizedEllipsoidsDataset [{os.path.basename(rasterized_dir)}]: "
              f"{len(self)} eps, horizon={self.horizon}")
        print(f"  xy_min={self.xy_min}  xy_max={self.xy_max}")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.occ_maps[idx]

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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    import PIL.Image
    img = PIL.Image.open(buf).convert("RGB")
    arr = np.array(img)
    return torch.tensor(arr).permute(2, 0, 1)


def build_preview_figure(
    ema_model, ve, ds_train,
    fix_xs, fix_xg, fix_occ, fix_ells_world,
    step, device,
):
    """Sample 4 trajectories from the eval set and return a matplotlib figure."""
    ema_model.eval()
    n = fix_xs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        f"Step {step:,}  —  DPM-Solver-1 CFG  "
        f"(w={GUIDANCE_SCALE}, {N_SAMPLE_STEPS} steps)  [eval set]",
        fontsize=11,
    )

    with torch.no_grad():
        for j, ax in enumerate(axes):
            samp = dpm_solver_1_cfg_sample(
                ema_model, ve,
                x_start=fix_xs[j:j+1], x_goal=fix_xg[j:j+1],
                occ_map=fix_occ[j:j+1],
                T_steps=ds_train.horizon,
                n_steps=N_SAMPLE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                device=device,
            )   # [1, T, 2]  — in training-norm space

            world = ds_train.unnormalize(samp.cpu())
            xs_w  = ds_train.unnormalize(fix_xs[j:j+1].cpu())
            xg_w  = ds_train.unnormalize(fix_xg[j:j+1].cpu())

            _draw_maze(ax)
            for row in fix_ells_world[j]:
                cx, cy, a, b = row
                if a > 1e-4 or b > 1e-4:
                    _draw_ellipsoid(ax, cx, cy, a, b)

            ax.plot(world[0, :, 0], world[0, :, 1], lw=1.2, color="steelblue", alpha=0.85)
            ax.scatter(xs_w[0, 0], xs_w[0, 1], c="green", s=60, zorder=5)
            ax.scatter(xg_w[0, 0], xg_w[0, 1], c="red",   s=60, zorder=5)
            ax.set_title(f"eval pair {j}")
            ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LocalFeats Rasterized Diffuser")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in CKPT_DIR")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help=f"Total training steps (default: {TOTAL_STEPS})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training dataset + DataLoader
    # -----------------------------------------------------------------------
    ds = RasterizedEllipsoidsDataset(TRAIN_RASTERIZED_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=(device == "cuda"))
    T_STEPS = ds.horizon
    print(f"T_STEPS={T_STEPS}, batches/epoch={len(dl)}")

    # -----------------------------------------------------------------------
    # Eval dataset (held-out; used for preview sampling only)
    # -----------------------------------------------------------------------
    ds_eval = RasterizedEllipsoidsDataset(EVAL_RASTERIZED_DIR, T_steps=T_STEPS)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    MAP_H, MAP_W = ds.occ_maps.shape[-2], ds.occ_maps.shape[-1]

    score_net = TemporalUnet(
        state_dim=2,
        T_steps=T_STEPS,
        unet_input_dim=UNET_INPUT_DIM,
        dim_mults=DIM_MULTS,
        local_dim=LOCAL_DIM,
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
    # Fixed preview samples — drawn from eval dataset, renormalized to training space
    # -----------------------------------------------------------------------
    rng_fix  = np.random.default_rng(42)
    fix_idx  = rng_fix.choice(len(ds_eval), size=10, replace=False).tolist()

    fix_trajs_eval, fix_occ_eval = zip(*[ds_eval[i] for i in fix_idx])
    fix_trajs_eval = torch.stack(fix_trajs_eval)   # [10, T, 2] — eval normalised space
    fix_occ        = torch.stack(fix_occ_eval).to(device)    # [10, 1, H, W]
    fix_ells_world = ds_eval.ellipsoids_world[fix_idx]        # [10, n_ell, 4] numpy

    # Renormalize trajectories from eval space to training space
    fix_trajs_world = ds_eval.unnormalize(fix_trajs_eval)
    fix_trajs       = ds.normalize(fix_trajs_world)

    fix_xs = fix_trajs[:, 0, :].to(device)
    fix_xg = fix_trajs[:, -1, :].to(device)

    # -----------------------------------------------------------------------
    # TensorBoard writer
    # -----------------------------------------------------------------------
    writer = SummaryWriter(log_dir=RUNS_DIR)
    print(f"\nTensorBoard logs: {RUNS_DIR}")
    print("  tensorboard --logdir EllipsoidalCBFSampling/runs")
    print("  http://localhost:6006\n")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    import itertools
    loader_iter   = itertools.cycle(dl)
    total_steps   = args.steps
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
        traj, occ_map = next(loader_iter)
        traj    = traj.to(device)
        occ_map = occ_map.to(device)

        xs = traj[:, 0, :]
        xg = traj[:, -1, :]

        loss, info = ve(traj, xs, xg, occ_map, p_uncond=P_UNCOND)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        if step >= EMA_START_STEP and step % EMA_UPDATE_EVERY == 0:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), score_net.parameters()):
                    p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)
                
                for b_ema, b in zip(ema_model.buffers(), score_net.buffers()):
                    b_ema.data.copy_(b.data)

        loss_val = info["loss"]
        loss_history.append(loss_val)
        recent_losses.append(loss_val)

        if step % LOG_EVERY == 0:
            avg_loss   = float(np.mean(recent_losses))
            sigma_mean = float(info["sigma_mean"])
            recent_losses.clear()

            writer.add_scalar("train/loss",       avg_loss,   step)
            writer.add_scalar("train/sigma_mean", sigma_mean, step)

            elapsed    = time.time() - t0
            steps_done = step - start_step
            sps        = steps_done / elapsed if elapsed > 0 else 0.0

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                sigma=f"{sigma_mean:.3f}",
                sps=f"{sps:.1f}",
            )

        if step % PREVIEW_EVERY == 0:
            fig = build_preview_figure(
                ema_model, ve, ds,
                fix_xs, fix_xg, fix_occ, fix_ells_world,
                step, device,
            )
            img_tensor = _fig_to_tensor(fig)
            writer.add_image("samples/trajectories", img_tensor, step)
            plt.close(fig)
            score_net.train()

        if step % SAVE_EVERY == 0:
            _save_checkpoint(
                step, score_net, ema_model, optimizer,
                loss_history, T_STEPS, MAP_H, MAP_W, ds,
            )
            writer.flush()

    _save_checkpoint(step, score_net, ema_model, optimizer,
                     loss_history, T_STEPS, MAP_H, MAP_W, ds)
    writer.close()
    pbar.close()
    print("Training complete.")


def _save_checkpoint(step, score_net, ema_model, optimizer,
                     loss_history, T_STEPS, MAP_H, MAP_W, ds):
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
                "local_dim":      LOCAL_DIM,
                "sigma_min":      SIGMA_MIN,
                "sigma_max":      SIGMA_MAX,
                "n_levels":       N_LEVELS,
                "map_h":          MAP_H,
                "map_w":          MAP_W,
                "p_uncond":       P_UNCOND,
                "guidance_scale": GUIDANCE_SCALE,
                "train_rasterized_dir": TRAIN_RASTERIZED_DIR,
                "eval_rasterized_dir":  EVAL_RASTERIZED_DIR,
                "xy_min":         ds.xy_min.tolist(),
                "xy_max":         ds.xy_max.tolist(),
            },
        },
        os.path.join(CKPT_DIR, f"ve_unet_largemaze_localfeats_step{step}.pt"),
    )
    print(f"  [step {step:,}] checkpoint saved")


if __name__ == "__main__":
    main()
