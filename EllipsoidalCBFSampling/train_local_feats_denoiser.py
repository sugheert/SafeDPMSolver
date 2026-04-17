"""
train_local_feats_denoiser.py — Standalone training script for the LargeMaze
ellipsoidal obstacles diffusion model (LightweightLocalEncoder / early-fusion).

Mirrors train_large_ellipsoids_diffuser.ipynb with:
  - TensorBoard metrics  (Loss/step, Sigma/mean, Loss/avg_N, Preview/scenes)
  - tqdm progress bar
  - --resume support (resumes from a step-stamped .pt checkpoint)
  - 5 fixed preview scenes generated once at startup (each with ≥3 ellipsoids)

Usage
-----
  # Fresh run
  python train_local_feats_denoiser.py

  # Smoke test (fast)
  python train_local_feats_denoiser.py \
      --total_steps 50 --batch_size 4 \
      --log_every 10 --preview_every 10 --save_every 25

  # Resume from checkpoint
  python train_local_feats_denoiser.py \
      --resume checkpoints/local_feats/ve_unet_largemaze_local_feats_step0000025.pt \
      --total_steps 100
"""

# ── matplotlib backend MUST be set before any pyplot import ──────────────────
import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import itertools
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # EllipsoidalCBFSampling/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                  # SDPMSP_Clone/

for _p in [PROJECT_ROOT, SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_traj_gen_dir = os.path.join(PROJECT_ROOT, 'Diffuser', 'TrajectoryDatasetGeneration')
if _traj_gen_dir not in sys.path:
    sys.path.insert(0, _traj_gen_dir)

_mpd_root = 'C:\\Users\\Owner\\SAFE_DIFFUSION\\mpd-public'
for _p in [_mpd_root, os.path.join(_mpd_root, 'deps', 'torch_robotics')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Project imports ───────────────────────────────────────────────────────────
import minari

from EllipsoidalCBFSampling.models.score_net_ellipsoids import TemporalUnet
from EllipsoidalCBFSampling.models.ve_diffusion_ellipsoids import VEDiffusion
from EllipsoidalCBFSampling.models.samplers_ellipsoids_cfg import dpm_solver_1_cfg_sample

from trajectory_generator import EnvLargeMazeRandomEllipsoids2D
from torch_robotics.robots import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask


# =============================================================================
# Section 2 — Dataset
# =============================================================================

class EllipsoidsMinariDataset(Dataset):
    """
    Wraps a Minari dataset that stores 2D trajectories with ellipsoidal obstacles.

    Each item returns:
        traj       : [T_steps, 2]          float32  — x,y normalised to [-1, 1]
        ellipsoids : [max_ellipsoids, 4]   float32  — (cx, cy, a, b); zeros for absent
    """

    def __init__(self, dataset_id, local_data_path, max_ellipsoids=5, T_steps=None):
        os.environ['MINARI_DATASETS_PATH'] = os.path.abspath(local_data_path)
        ds = minari.load_dataset(dataset_id)

        trajs      = []
        ellipsoids = []

        for ep in ds.iterate_episodes():
            obs = ep.observations
            trajs.append(obs[:, :2].astype(np.float32))

            centers = np.array(ep.infos['ellipsoids_centers'][0], dtype=np.float32)
            radii   = np.array(ep.infos['ellipsoids_radii'][0],   dtype=np.float32)
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

        print(f"EllipsoidsMinariDataset: {len(self)} episodes, horizon={self.horizon}, "
              f"ellipsoids/ep={max_ellipsoids}")
        print(f"  xy_min={self.xy_min}, xy_max={self.xy_max}")

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


# =============================================================================
# Section 3 — Maze drawing helpers
# =============================================================================

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


def draw_maze(ax):
    for i, row in enumerate(_MAZE_MAP):
        for j, cell in enumerate(row):
            if cell == 1:
                cx = -5.5 + j * 1.0
                cy =  4.0 - i * 1.0
                rect = plt.Rectangle(
                    (cx - 0.5, cy - 0.5), 1.0, 1.0,
                    linewidth=0, facecolor='dimgray', alpha=1.0,
                )
                ax.add_patch(rect)
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-4.5, 4.5)


def draw_ellipsoid(ax, cx, cy, a, b, color='gray', alpha=0.4):
    t = np.linspace(0, 2 * np.pi, 200)
    x = cx + a * np.sign(np.cos(t)) * np.sqrt(np.abs(np.cos(t)))
    y = cy + b * np.sign(np.sin(t)) * np.sqrt(np.abs(np.sin(t)))
    polygon = plt.Polygon(np.column_stack([x, y]),
                          facecolor=color, edgecolor=color, alpha=alpha)
    ax.add_patch(polygon)


# =============================================================================
# Section 4 — Random scene generation
# =============================================================================

_tensor_args_cpu = {'device': 'cpu', 'dtype': torch.float32}


def generate_random_scene(max_obstacles=5, dist_factor=0.25):
    """
    Sample a random LargeMaze scene with up to max_obstacles ellipsoidal obstacles
    and a collision-free start/goal pair.

    Returns
    -------
    ell_world : np.ndarray  [max_obstacles, 4]  (cx, cy, a, b) world coords
    start_pos : torch.Tensor [2]
    goal_pos  : torch.Tensor [2]
    """
    env   = EnvLargeMazeRandomEllipsoids2D(tensor_args=_tensor_args_cpu,
                                           max_obstacles=max_obstacles)
    robot = RobotPointMass(q_limits=env.limits, tensor_args=_tensor_args_cpu)
    task  = PlanningTask(env=env, robot=robot,
                         obstacle_cutoff_margin=0.08,
                         tensor_args=_tensor_args_cpu)

    min_dist = torch.norm(env.limits[1] - env.limits[0]) * dist_factor

    start_pos = goal_pos = None
    for _ in range(500):
        q_free = task.random_coll_free_q(n_samples=2)
        sp, gp = q_free[0], q_free[1]
        if torch.norm(sp - gp) > min_dist:
            start_pos, goal_pos = sp, gp
            break

    if start_pos is None:
        start_pos, goal_pos = q_free[0], q_free[1]

    centers = env.ellipsoids_centers
    radii   = env.ellipsoids_radii
    ell = np.concatenate([centers, radii], axis=-1).astype(np.float32)
    N = ell.shape[0]
    if N < max_obstacles:
        ell = np.concatenate([ell, np.zeros((max_obstacles - N, 4), dtype=np.float32)], axis=0)
    else:
        ell = ell[:max_obstacles]

    return ell, start_pos, goal_pos


def _generate_scene_with_min_obstacles(max_obstacles, min_obstacles, max_retries=20):
    """Retry until at least min_obstacles non-zero ellipsoids are placed."""
    ell_np = sp = gp = None
    n_nonzero = 0
    for _ in range(max_retries):
        ell_np, sp, gp = generate_random_scene(max_obstacles=max_obstacles)
        n_nonzero = int((np.abs(ell_np).sum(axis=1) > 1e-6).sum())
        if n_nonzero >= min_obstacles:
            return ell_np, sp, gp
    print(f"WARNING: only {n_nonzero} obstacles after {max_retries} retries; using anyway.")
    return ell_np, sp, gp


# =============================================================================
# Section 5 — Preview figure (pure function, no side-effects)
# =============================================================================

def make_preview_figure(ema_model, ve, preview_scenes, ds, device,
                        step, guidance_scale, n_sample_steps, T_steps):
    """
    Build a 1×5 matplotlib Figure showing DPM-Solver-1 CFG samples for the 5
    fixed preview scenes. Returns the Figure; caller must plt.close() it.
    """
    fig, axes = plt.subplots(1, len(preview_scenes),
                             figsize=(5 * len(preview_scenes), 5))
    fig.suptitle(
        f'Step {step:,}  —  DPM-Solver-1 CFG (w={guidance_scale}, {n_sample_steps} steps)',
        fontsize=11,
    )

    ema_model.eval()
    with torch.no_grad():
        for j, (ax, (ell_np, xs_pre, xg_pre)) in enumerate(zip(axes, preview_scenes)):
            xs_j  = xs_pre.to(device)
            xg_j  = xg_pre.to(device)
            ell_j = torch.tensor(ell_np, dtype=torch.float32).unsqueeze(0).to(device)

            samp = dpm_solver_1_cfg_sample(
                ema_model, ve,
                x_start=xs_j, x_goal=xg_j,
                ellipsoids=ell_j,
                T_steps=T_steps,
                n_steps=n_sample_steps,
                guidance_scale=guidance_scale,
                device=device,
            )

            world = ds.unnormalize(samp.cpu())
            xs_w  = ds.unnormalize(xs_pre)
            xg_w  = ds.unnormalize(xg_pre)

            draw_maze(ax)
            for row in ell_np:
                cx, cy, a, b = row
                if a > 1e-4 or b > 1e-4:
                    draw_ellipsoid(ax, cx, cy, a, b)

            ax.plot(world[0, :, 0], world[0, :, 1], lw=1.2, color='steelblue', alpha=0.85)
            ax.scatter(xs_w[0, 0], xs_w[0, 1], c='green', s=60, zorder=5)
            ax.scatter(xg_w[0, 0], xg_w[0, 1], c='red',   s=60, zorder=5)
            ax.set_title(f'fixed scene {j + 1}')
            ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# =============================================================================
# Section 6 — Checkpoint helpers
# =============================================================================

def save_checkpoint(path, step, score_net, ema_model, optimizer,
                    loss_history, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step':         step,
        'score_net':    score_net.state_dict(),
        'ema_model':    ema_model.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'loss_history': loss_history,
        'config':       config,
    }, path)
    print(f'  checkpoint saved → {path}')


# =============================================================================
# Section 7 — Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Train LargeMaze ellipsoid diffusion model (local-feats / early-fusion).'
    )

    # Dataset
    p.add_argument('--dataset_id',      default='custom/largemaze_ellipsoids-v1')
    p.add_argument('--local_data_path', default='../Diffuser/TrajectoryDatasetGeneration/data',
                   help='Path to Minari data dir (relative to script or absolute)')
    p.add_argument('--max_ellipsoids',  type=int,   default=5)

    # Model architecture
    p.add_argument('--unet_input_dim',  type=int,   default=32)
    p.add_argument('--dim_mults',       default='1,2,4',
                   help='Comma-separated channel multipliers, e.g. "1,2,4"')
    p.add_argument('--local_hidden_dim',type=int,   default=128)
    p.add_argument('--local_dim',       type=int,   default=64)

    # VE noise schedule
    p.add_argument('--sigma_min',       type=float, default=0.01)
    p.add_argument('--sigma_max',       type=float, default=10.0)
    p.add_argument('--n_levels',        type=int,   default=1000)

    # CFG
    p.add_argument('--p_uncond',        type=float, default=0.1)
    p.add_argument('--guidance_scale',  type=float, default=1.0)

    # Training
    p.add_argument('--total_steps',     type=int,   default=100_000)
    p.add_argument('--batch_size',      type=int,   default=128)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--ema_decay',       type=float, default=0.995)
    p.add_argument('--ema_start_step',  type=int,   default=1_000)
    p.add_argument('--ema_update_every',type=int,   default=10)

    # Sampling (for preview only)
    p.add_argument('--n_sample_steps',  type=int,   default=25)

    # Logging / checkpointing
    p.add_argument('--log_every',       type=int,   default=500)
    p.add_argument('--preview_every',   type=int,   default=500)
    p.add_argument('--save_every',      type=int,   default=10_000)
    p.add_argument('--ckpt_dir',        default='checkpoints/local_feats',
                   help='Checkpoint directory (relative to script or absolute)')
    p.add_argument('--tb_dir',          default='runs/local_feats',
                   help='TensorBoard root directory (relative to script or absolute)')

    # Resume
    p.add_argument('--resume',          default=None,
                   help='Path to a .pt checkpoint to resume from')

    # Device
    p.add_argument('--device',          default='auto',
                   help='"auto" selects cuda if available, else cpu')

    return p.parse_args()


# =============================================================================
# Section 8 — Main
# =============================================================================

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'device: {device}')

    dim_mults = tuple(int(x) for x in args.dim_mults.split(','))

    # ── Resolve relative paths against the script directory ──────────────────
    def _abs(path):
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(SCRIPT_DIR, path))

    local_data_path = _abs(args.local_data_path)
    ckpt_dir        = _abs(args.ckpt_dir)
    tb_dir          = _abs(args.tb_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = EllipsoidsMinariDataset(
        args.dataset_id, local_data_path, max_ellipsoids=args.max_ellipsoids,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    T_STEPS = ds.horizon
    print(f'T_STEPS={T_STEPS}, batches per epoch={len(dl)}')

    # ── Model (fresh or resumed) ──────────────────────────────────────────────
    if args.resume:
        print(f'Resuming from {args.resume} ...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg  = ckpt['config']

        score_net = TemporalUnet(
            state_dim=2,
            T_steps=cfg['T_steps'],
            unet_input_dim=cfg['unet_input_dim'],
            dim_mults=tuple(cfg['dim_mults']),
            local_hidden_dim=cfg['local_hidden_dim'],
            local_dim=cfg['local_dim'],
        ).to(device)

        ve = VEDiffusion(
            score_net,
            sigma_min=cfg['sigma_min'],
            sigma_max=cfg['sigma_max'],
            n_levels=cfg['n_levels'],
        ).to(device)

        ema_model = copy.deepcopy(score_net).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

        score_net.load_state_dict(ckpt['score_net'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        start_step   = ckpt['step']
        loss_history = ckpt['loss_history']
        print(f'Resumed at step {start_step:,}')

        # Use arch config from checkpoint (not CLI) to avoid silent shape mismatches
        unet_input_dim   = cfg['unet_input_dim']
        local_hidden_dim = cfg['local_hidden_dim']
        local_dim        = cfg['local_dim']
        sigma_min        = cfg['sigma_min']
        sigma_max        = cfg['sigma_max']
        n_levels         = cfg['n_levels']
        max_ellipsoids   = cfg.get('max_ellipsoids', args.max_ellipsoids)
        T_STEPS          = cfg['T_steps']

    else:
        score_net = TemporalUnet(
            state_dim=2,
            T_steps=T_STEPS,
            unet_input_dim=args.unet_input_dim,
            dim_mults=dim_mults,
            local_hidden_dim=args.local_hidden_dim,
            local_dim=args.local_dim,
        ).to(device)

        ve = VEDiffusion(
            score_net,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            n_levels=args.n_levels,
        ).to(device)

        ema_model = copy.deepcopy(score_net).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

        start_step   = 0
        loss_history = []

        unet_input_dim   = args.unet_input_dim
        local_hidden_dim = args.local_hidden_dim
        local_dim        = args.local_dim
        sigma_min        = args.sigma_min
        sigma_max        = args.sigma_max
        n_levels         = args.n_levels
        max_ellipsoids   = args.max_ellipsoids

    n_params = sum(p.numel() for p in score_net.parameters())
    print(f'TemporalUnet params: {n_params:,}')

    # Config dict saved into every checkpoint
    config = {
        'T_steps':          T_STEPS,
        'unet_input_dim':   unet_input_dim,
        'dim_mults':        list(dim_mults),
        'sigma_min':        sigma_min,
        'sigma_max':        sigma_max,
        'n_levels':         n_levels,
        'local_hidden_dim': local_hidden_dim,
        'local_dim':        local_dim,
        'max_ellipsoids':   max_ellipsoids,
        'p_uncond':         args.p_uncond,
        'guidance_scale':   args.guidance_scale,
        'dataset_id':       args.dataset_id,
        'xy_min':           ds.xy_min.tolist(),
        'xy_max':           ds.xy_max.tolist(),
    }

    # ── Fixed preview scenes ──────────────────────────────────────────────────
    print('Generating 5 fixed preview scenes (min 3 obstacles each) ...')
    preview_scenes = []
    for _i in range(5):
        _ell, _sp, _gp = _generate_scene_with_min_obstacles(
            max_obstacles=max_ellipsoids, min_obstacles=3,
        )
        preview_scenes.append((
            _ell,
            ds.normalize(_sp.unsqueeze(0)),   # [1, 2] normalised CPU
            ds.normalize(_gp.unsqueeze(0)),   # [1, 2] normalised CPU
        ))
        _n = int((np.abs(_ell).sum(axis=1) > 1e-6).sum())
        print(f'  scene {_i + 1}: {_n} obstacles')

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_log_dir = os.path.join(tb_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'TensorBoard log dir: {tb_log_dir}')

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    loader_iter = itertools.cycle(dl)

    pbar = tqdm(
        range(start_step + 1, args.total_steps + 1),
        initial=start_step,
        total=args.total_steps,
        desc='Training',
        unit='step',
        dynamic_ncols=True,
    )

    for step in pbar:
        score_net.train()

        traj, ellipsoids = next(loader_iter)
        traj       = traj.to(device)
        ellipsoids = ellipsoids.to(device)
        xs = traj[:, 0, :]
        xg = traj[:, -1, :]

        loss, info = ve(traj, xs, xg, ellipsoids, p_uncond=args.p_uncond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        if step >= args.ema_start_step and step % args.ema_update_every == 0:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), score_net.parameters()):
                    p_ema.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        loss_history.append(info['loss'])

        # TensorBoard — scalar per step
        writer.add_scalar('Loss/step',  info['loss'],       step)
        writer.add_scalar('Sigma/mean', info['sigma_mean'], step)

        # Smoothed loss + tqdm postfix
        if step % args.log_every == 0:
            avg = float(np.mean(loss_history[-args.log_every:]))
            writer.add_scalar(f'Loss/avg_{args.log_every}', avg, step)
            pbar.set_postfix(loss=f'{avg:.4f}', sigma=f'{info["sigma_mean"]:.3f}')

        # Preview figure to TensorBoard
        if step % args.preview_every == 0:
            fig = make_preview_figure(
                ema_model, ve, preview_scenes, ds, device,
                step, args.guidance_scale, args.n_sample_steps, T_STEPS,
            )
            writer.add_figure('Preview/scenes', fig, global_step=step)
            plt.close(fig)
            score_net.train()

        # Checkpoint
        if step % args.save_every == 0:
            ckpt_filename = f've_unet_largemaze_local_feats_step{step:07d}.pt'
            ckpt_path     = os.path.join(ckpt_dir, ckpt_filename)
            save_checkpoint(ckpt_path, step, score_net, ema_model,
                            optimizer, loss_history, config)

    writer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
