import sys, os, copy, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Add the parent of SafeDPMSolver/ so that 'import SafeDPMSolver' works
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SafeDPMSolver.models.score_net import ScoreNet
from SafeDPMSolver.models.ve_diffusion import VEDiffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')\n# ---- Trajectory shape ----
T_STEPS = 64        # waypoints per trajectory

# ---- Model (Temporal U-Net, MPD-style) ----
UNET_INPUT_DIM        = 32
DIM_MULTS             = (1, 2, 4)
TIME_EMB_DIM          = 32
CONDITIONING_EMB_DIM  = 4      # cat(x_start, x_goal) = 4-dim

# ---- Noise schedule (VE-SDE) ----
SIGMA_MIN  = 0.01
SIGMA_MAX  = 10.0
N_LEVELS   = 1000              # discrete levels during training

# ---- Training (MPD-matched) ----
BATCH_SIZE       = 128
LR               = 3e-4
EMA_DECAY        = 0.995
EMA_START_STEP   = 1_000
EMA_UPDATE_EVERY = 10
TOTAL_STEPS      = 100_000
LOG_EVERY        = 1_000       # log + visualise every N steps

# ---- Sampling ----
N_SAMPLE_STEPS = 25            # MPD uses 25
N_SAMPLES_VIS  = 16            # trajectories for waypoint scatter
N_TRAJ_VIS     = 5             # trajectories drawn as lines\n# ---- Path to MPD data (edit if you moved it) ----
MPD_DATA_ROOT = r'C:\Users\Owner\Downloads\MPD\mpd-public\data_trajectories'
MPD_ENV       = 'EnvSimple2D-RobotPointMass'


class MPDTrajectoryDataset(Dataset):
    """
    Loads MPD pre-generated trajectories as complete sequences.

    Each trajs-free.pt file: [n_trajs, 64, 4]  (x, y, vx, vy).
    We use only (x, y) and return one full trajectory [T_STEPS, 2] per sample,
    along with its start [2] and goal [2].
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
                trajs_l.append(traj)              # [64, 2]
                starts_l.append(traj[0])          # [2]
                goals_l.append(traj[-1])          # [2]

        self.trajs  = torch.stack(trajs_l)    # [N, 64, 2]
        self.starts = torch.stack(starts_l)   # [N, 2]
        self.goals  = torch.stack(goals_l)    # [N, 2]

        print(f'[MPDTrajectoryDataset]  env={os.path.basename(env_dir)}')
        print(f'  {len(self.trajs):,} trajectories  |  shape per sample: {tuple(self.trajs[0].shape)}')

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.starts[idx], self.goals[idx]


dataset = MPDTrajectoryDataset(
    env_dir=os.path.join(MPD_DATA_ROOT, MPD_ENV),
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Quick look
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel 1: all waypoints (unrolled) to reveal obstacle voids
all_pts = dataset.trajs.reshape(-1, 2).numpy()
axes[0].scatter(all_pts[:, 0], all_pts[:, 1],
                s=0.3, alpha=0.07, c='steelblue', rasterized=True)
axes[0].set_xlim(-1.1, 1.1); axes[0].set_ylim(-1.1, 1.1)
axes[0].set_aspect('equal')
axes[0].set_title(f'Training data distribution — {MPD_ENV}', fontsize=10)
axes[0].grid(True, alpha=0.2)

# Panel 2: 8 example trajectories
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for i, color in enumerate(colors):
    traj = dataset.trajs[i * 50].numpy()
    axes[1].plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.8, linewidth=1.0)
    axes[1].scatter(*traj[0],  s=70,  color=color, edgecolors='k', zorder=5, marker='s')
    axes[1].scatter(*traj[-1], s=90,  color=color, edgecolors='k', zorder=5, marker='*')
axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-1.1, 1.1)
axes[1].set_aspect('equal')
axes[1].set_title('Example trajectories (squares=start, stars=goal)', fontsize=10)
axes[1].grid(True, alpha=0.2)

fig.suptitle('MPD Dataset — full trajectories in 2D workspace', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# Background points for later visualisation (reused across cells)
bg_idx = np.random.default_rng(0).choice(len(all_pts), size=20_000, replace=False)
bg_pts = all_pts[bg_idx]\nenv_func_path =  os.path.join(os.path.dirname(os.getcwd()), 'environments')
print(f'Appending to sys.path: {env_func_path}')
sys.path.append(env_func_path)

from circles_obstacles import load_mpd_deps, get_circles_from_env

load_mpd_deps()
ENV_CIRCLES = get_circles_from_env()\nfrom SafeDPMSolver.models.score_net import TemporalUnet

score_net = TemporalUnet(
    state_dim=2,
    T_steps=T_STEPS,
    unet_input_dim=UNET_INPUT_DIM,
    dim_mults=DIM_MULTS,
    time_emb_dim=TIME_EMB_DIM,
    conditioning_embed_dim=CONDITIONING_EMB_DIM,
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

n_params = sum(p.numel() for p in score_net.parameters())
cond_dim = TIME_EMB_DIM + CONDITIONING_EMB_DIM
print(f'TemporalUnet  |  {n_params:,} parameters')
print(f'U-Net dims    |  state={2} → {[UNET_INPUT_DIM * m for m in DIM_MULTS]}')
print(f'cond_dim      |  {TIME_EMB_DIM} (sigma) + {CONDITIONING_EMB_DIM} (start+goal) = {cond_dim}')
print(f'Horizon T     |  {T_STEPS} waypoints  |  device: {device}')\n@torch.no_grad()
def ancestral_sample(
    model,
    ve_diffusion,
    x_start,              # [2] or [B, 2]
    x_goal,               # [2] or [B, 2]
    T_steps=64,
    n_steps=200,
    device='cpu',
    return_trajectory=False,
):
    """
    Ancestral sampler (Euler-Maruyama) for VE-SDE over full trajectories.

    Denoises a [B, T_steps, 2] tensor from pure noise down to a clean trajectory.

    Returns:
        x_final    : [B, T_steps, 2]
        trajectory : [n_steps+1, B, T_steps, 2]  (only if return_trajectory=True)
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start = x_start.to(device)
    x_goal  = x_goal.to(device)

    sigmas = ve_diffusion.sigmas.to(device)
    N = ve_diffusion.n_levels
    indices   = torch.linspace(N, 0, n_steps + 1).long().clamp(0, N)
    sigma_seq = sigmas[indices]   # [n_steps+1], decreasing

    # Start from pure noise over the full trajectory
    x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    trajectory = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        sig_cur  = sigma_seq[i]
        sig_next = sigma_seq[i + 1]

        if sig_next <= 0:
            break

        sig_batch = sig_cur.expand(B)
        eps_pred  = model(x, sig_batch, x_start, x_goal)   # [B, T, 2]

        delta_sig2 = sig_cur**2 - sig_next**2
        x = x - (delta_sig2 / sig_cur) * eps_pred

        if i < n_steps - 1:
            x = x + torch.sqrt(delta_sig2) * torch.randn_like(x)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=0)   # [B,T,2], [steps+1,B,T,2]
    return x   # [B, T, 2]


print('Ancestral sampler defined.')\n@torch.no_grad()
def dpm_solver_1_sample(
    model,
    ve_diffusion,
    x_start,              # [2] or [B, 2]
    x_goal,               # [2] or [B, 2]
    T_steps=64,
    n_steps=25,
    device='cpu',
):
    """
    DPM-Solver-1 (first-order deterministic ODE) for VE-SDE.

    x_{i+1} = x_i - (sigma_i - sigma_{i+1}) * eps_theta(x_i, sigma_i)

    Returns:
        x_final : [B, T_steps, 2]
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start = x_start.to(device)
    x_goal  = x_goal.to(device)

    sigmas = ve_diffusion.sigmas.to(device)
    N = ve_diffusion.n_levels
    indices   = torch.linspace(N, 0, n_steps + 1).long().clamp(0, N)
    sigma_seq = sigmas[indices]   # [n_steps+1], monotonically decreasing

    x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    for i in range(n_steps):
        sig_cur  = sigma_seq[i]
        sig_next = sigma_seq[i + 1]

        sig_batch = sig_cur.expand(B)
        eps_pred  = model(x, sig_batch, x_start, x_goal)   # [B, T, 2]

        x = x - (sig_cur - sig_next) * eps_pred             # deterministic ODE step

    return x   # [B, T_steps, 2]


print('DPM-Solver-1 sampler defined.')\nimport itertools

rng_fix     = np.random.default_rng(7)
fixed_idx   = int(rng_fix.integers(0, len(dataset.trajs)))
fixed_start = dataset.starts[fixed_idx].to(device)
fixed_goal  = dataset.goals[fixed_idx].to(device)

loss_history  = []   # (step, loss) pairs
step          = 0
loader_iter   = itertools.cycle(loader)

print(f'Training for {TOTAL_STEPS:,} steps  |  batch={BATCH_SIZE}  |  lr={LR}  |  device={device}')
print(f'EMA: decay={EMA_DECAY}  start={EMA_START_STEP}  update_every={EMA_UPDATE_EVERY}')
print('=' * 70)

while step < TOTAL_STEPS:
    # ---- Training step ----
    score_net.train()
    x, xs, xg = next(loader_iter)
    x, xs, xg = x.to(device), xs.to(device), xg.to(device)

    loss, info = ve.loss(x, xs, xg)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
    optimizer.step()

    # ---- EMA update ----
    if step >= EMA_START_STEP and step % EMA_UPDATE_EVERY == 0:
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), score_net.parameters()):
                p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

    step += 1

    # ---- Logging & visualisation ----
    if step % LOG_EVERY == 0 or step == 1:
        loss_history.append((step, info['loss']))
        ema_model.eval()

        xs_b = fixed_start.unsqueeze(0).expand(N_SAMPLES_VIS, -1)
        xg_b = fixed_goal.unsqueeze(0).expand(N_SAMPLES_VIS, -1)

        samples = dpm_solver_1_sample(
            ema_model, ve, xs_b, xg_b,
            T_steps=T_STEPS, n_steps=N_SAMPLE_STEPS, device=device,
        ).cpu().numpy()   # [N_SAMPLES_VIS, T, 2]

        xs_t = fixed_start.unsqueeze(0).expand(N_TRAJ_VIS, -1)
        xg_t = fixed_goal.unsqueeze(0).expand(N_TRAJ_VIS, -1)
        final_trajs, _ = ancestral_sample(
            ema_model, ve, xs_t, xg_t,
            T_steps=T_STEPS, n_steps=N_SAMPLE_STEPS, device=device,
            return_trajectory=True,
        )
        final_trajs = final_trajs.cpu().numpy()   # [N_TRAJ_VIS, T, 2]

        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        s_np = fixed_start.cpu().numpy()
        g_np = fixed_goal.cpu().numpy()
        colors = plt.cm.tab10(np.linspace(0, 1, N_TRAJ_VIS))

        # Panel 1: loss curve
        steps_l, losses_l = zip(*loss_history)
        axes[0].plot(steps_l, losses_l, 'b-', linewidth=1.5)
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('DSM Loss')
        axes[0].set_title(f'Training Loss (step {step:,})')
        axes[0].grid(True, alpha=0.3)

        # Panel 2: waypoint scatter
        scat_pts = samples.reshape(-1, 2)
        axes[1].scatter(bg_pts[:, 0], bg_pts[:, 1],
                        s=0.3, alpha=0.07, c='gray', rasterized=True)
        axes[1].scatter(scat_pts[:, 0], scat_pts[:, 1],
                        s=4, alpha=0.4, c='crimson', edgecolors='none')
        axes[1].scatter(*s_np, s=120, c='lime', edgecolors='black', zorder=5, marker='s')
        axes[1].scatter(*g_np, s=120, c='gold', edgecolors='black', zorder=5, marker='*')
        axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-1.1, 1.1)
        axes[1].set_aspect('equal')
        axes[1].set_title(f'Generated waypoints (step {step:,})')
        axes[1].grid(True, alpha=0.2)

        # Panel 3: trajectories as lines
        axes[2].scatter(bg_pts[:, 0], bg_pts[:, 1],
                        s=0.3, alpha=0.05, c='gray', rasterized=True)
        for j in range(N_TRAJ_VIS):
            traj_j = final_trajs[j]
            axes[2].plot(traj_j[:, 0], traj_j[:, 1], '-', color=colors[j], alpha=0.8, linewidth=1.2)
            axes[2].scatter(*traj_j[0],  s=40, color=colors[j], edgecolors='k', zorder=5, marker='s')
            axes[2].scatter(*traj_j[-1], s=50, color=colors[j], edgecolors='k', zorder=5, marker='*')
        axes[2].scatter(*s_np, s=120, c='lime', edgecolors='black', zorder=6, marker='s')
        axes[2].scatter(*g_np, s=120, c='gold', edgecolors='black', zorder=6, marker='*')
        axes[2].set_xlim(-1.1, 1.1); axes[2].set_ylim(-1.1, 1.1)
        axes[2].set_aspect('equal')
        axes[2].set_title(f'Generated trajectories (step {step:,})')
        axes[2].grid(True, alpha=0.2)

        fig.suptitle(
            f'Step {step:,}/{TOTAL_STEPS:,}  |  loss={info["loss"]:.4f}  |  σ_mean={info["sigma_mean"]:.3f}  |  {MPD_ENV}',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout(); plt.show()

print('\nTraining complete!')\nCKPT_PATH = os.path.join(os.path.dirname(os.getcwd()), 'checkpoints', 've_unet_circles_100k.pt')
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# ---- Save ----
torch.save({
    'step':            step,
    'score_net':       score_net.state_dict(),
    'ema_model':       ema_model.state_dict(),
    'optimizer':       optimizer.state_dict(),
    'loss_history':    loss_history,
    # model config — needed to reconstruct the architecture on load
    'model_cfg': dict(
        state_dim=2, T_steps=T_STEPS,
        unet_input_dim=UNET_INPUT_DIM, dim_mults=DIM_MULTS,
        time_emb_dim=TIME_EMB_DIM, conditioning_embed_dim=CONDITIONING_EMB_DIM,
    ),
    've_cfg': dict(
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, n_levels=N_LEVELS,
    ),
}, CKPT_PATH)
print(f'Saved checkpoint  →  {CKPT_PATH}  (step {step:,})')\n# ---- Load ----
CKPT_PATH = os.path.join(os.path.dirname(os.getcwd()), 'checkpoints', 've_unet_circles_100k.pt')

ckpt = torch.load(CKPT_PATH, map_location=device)

# Rebuild model from saved config
score_net = TemporalUnet(**ckpt['model_cfg']).to(device)
ve        = VEDiffusion(model=score_net, **ckpt['ve_cfg']).to(device)
ema_model = copy.deepcopy(score_net).to(device)
for p in ema_model.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.Adam(score_net.parameters(), lr=LR)

score_net.load_state_dict(ckpt['score_net'])
ema_model.load_state_dict(ckpt['ema_model'])
optimizer.load_state_dict(ckpt['optimizer'])
step         = ckpt['step']
loss_history = ckpt['loss_history']

print(f'Loaded checkpoint  ←  {CKPT_PATH}  (step {step:,})')
print(f'Resume training by re-running the train cell from step {step:,}')\nema_model.eval()

rng2      = np.random.default_rng(99)
traj_idxs = rng2.choice(len(dataset.trajs), size=4, replace=False)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for ax_i, tidx in enumerate(traj_idxs):
    xs_i = dataset.starts[int(tidx)].to(device)
    xg_i = dataset.goals[int(tidx)].to(device)

    xs_b = xs_i.unsqueeze(0).expand(8, -1)
    xg_b = xg_i.unsqueeze(0).expand(8, -1)

    samps = ancestral_sample(
        ema_model, ve, xs_b, xg_b,
        T_steps=T_STEPS, n_steps=N_SAMPLE_STEPS, device=device,
    ).cpu().numpy()   # [8, T, 2]

    colors4 = plt.cm.tab10(np.linspace(0, 1, 8))
    axes[ax_i].scatter(bg_pts[:, 0], bg_pts[:, 1],
                       s=0.2, alpha=0.06, c='gray', rasterized=True)
    for j in range(8):
        axes[ax_i].plot(samps[j, :, 0], samps[j, :, 1],
                        '-', color=colors4[j], alpha=0.7, linewidth=1.0)
        axes[ax_i].scatter(*samps[j, 0],  s=30, color=colors4[j], edgecolors='k', zorder=5, marker='s')
        axes[ax_i].scatter(*samps[j, -1], s=40, color=colors4[j], edgecolors='k', zorder=5, marker='*')

    s_np_i = xs_i.cpu().numpy()
    g_np_i = xg_i.cpu().numpy()
    axes[ax_i].scatter(*s_np_i, s=120, c='lime', edgecolors='black', zorder=6, marker='s')
    axes[ax_i].scatter(*g_np_i, s=120, c='gold', edgecolors='black', zorder=6, marker='*')
    axes[ax_i].set_xlim(-1.1, 1.1); axes[ax_i].set_ylim(-1.1, 1.1)
    axes[ax_i].set_aspect('equal')
    axes[ax_i].set_title(f'Context {ax_i + 1}', fontsize=10)
    axes[ax_i].grid(True, alpha=0.2)

fig.suptitle(f'8 generated trajectories per context — {MPD_ENV}',
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()\nema_model.eval()

rng3       = np.random.default_rng(42)
traj_idxs3 = rng3.choice(len(dataset.trajs), size=4, replace=False)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for ax_i, tidx in enumerate(traj_idxs3):
    xs_i = dataset.starts[int(tidx)].to(device)
    xg_i = dataset.goals[int(tidx)].to(device)

    xs_b = xs_i.unsqueeze(0).expand(8, -1)
    xg_b = xg_i.unsqueeze(0).expand(8, -1)

    samps = dpm_solver_1_sample(
        ema_model, ve, xs_b, xg_b,
        T_steps=T_STEPS, n_steps=5, device=device,
    ).cpu().numpy()   # [8, T, 2]

    colors4 = plt.cm.tab10(np.linspace(0, 1, 8))
    axes[ax_i].scatter(bg_pts[:, 0], bg_pts[:, 1],
                       s=0.2, alpha=0.06, c='gray', rasterized=True)
    for j in range(8):
        axes[ax_i].plot(samps[j, :, 0], samps[j, :, 1],
                        '-', color=colors4[j], alpha=0.7, linewidth=1.0)
        axes[ax_i].scatter(*samps[j, 0],  s=30, color=colors4[j], edgecolors='k', zorder=5, marker='s')
        axes[ax_i].scatter(*samps[j, -1], s=40, color=colors4[j], edgecolors='k', zorder=5, marker='*')

    s_np_i = xs_i.cpu().numpy()
    g_np_i = xg_i.cpu().numpy()
    axes[ax_i].scatter(*s_np_i, s=120, c='lime', edgecolors='black', zorder=6, marker='s')
    axes[ax_i].scatter(*g_np_i, s=120, c='gold', edgecolors='black', zorder=6, marker='*')
    axes[ax_i].set_xlim(-1.1, 1.1); axes[ax_i].set_ylim(-1.1, 1.1)
    axes[ax_i].set_aspect('equal')
    axes[ax_i].set_title(f'Context {ax_i + 1}', fontsize=10)
    axes[ax_i].grid(True, alpha=0.2)

fig.suptitle(f'DPM-Solver-1 — 8 trajectories per context — {MPD_ENV}',
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()\nimport sys
_cbf_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if _cbf_root not in sys.path:
    sys.path.insert(0, _cbf_root)

from SafeDPMSolver.CBF.trajectory_cbf import trajectory_cbf, grad_hXt_dXt


def cbf_control_term(
    Xt,           # [T, 2]  current trajectory (single sample, no batch dim)
    eps_pred,     # [T, 2]  predicted noise for this trajectory
    sig_cur,      # float   sigma at the current denoising step
    sig_next,     # float   sigma at the next denoising step
    sigma_dot,    # float   σ̇ at the current denoising step
    step_idx,     # int     loop counter: 0 = most noisy, n_steps-1 = cleanest
    n_steps,      # int     total denoising steps
    obstacles,    # [N, 3]  (px, py, r) in the same space as Xt
    k1=1.0,       # softmin temperature over obstacles  (m1 weights)
    k2=1.0,       # softmin temperature over waypoints  (m2 weights)
    alpha0=1.0,   # class-K linear gain  α(h) = alpha0 * h
    gamma_delta=0.0,
    eps=1e-8,
):
    """
    CBF control correction for a single trajectory [T, 2], eq. 70.

    Computes:
        rho       = max(0, step_idx - n_steps/2)      relaxation parameter
        h_val     = h(X_t)                            scalar CBF value
        grad_h    = grad_{X_t} h(X_t)                [T, 2]
        omega     = grad_h . (delta_sig * eps_pred) + alpha0 * h_val
        ctrl      = min(0, omega) / (||grad_h||^2 + rho^{-1} h^2) * grad_h

    Returns [T, 2].  The sampler subtracts this from x_new (eq. 70).
    When rho = 0 (first half of denoising) returns zeros — no correction applied.
    """
    rho = max(0.0, step_idx - n_steps / 2.0)   # relaxation γ  (zero for first half)
    if rho < eps:
        return torch.zeros_like(Xt)

    h_val  = trajectory_cbf(Xt, obstacles, k1, k2, gamma_delta)   # scalar
    grad_h = grad_hXt_dXt(Xt, obstacles, k1, k2, gamma_delta)     # [T, 2]

    # ω = ∇h^T · (σ_{i-1} − σ_i) ε_θ + α₀ h
    delta_sig = sig_cur - sig_next                                  # positive scalar
    omega     = (grad_h * (sigma_dot * eps_pred)).sum() + alpha0 * h_val

    omega_neg = torch.clamp(omega, max=0.0)                         # min(0, ω)
    denom     = (grad_h * grad_h).sum() + (1.0 / rho) * h_val ** 2 + eps

    return (omega_neg / denom) * grad_h * delta_sig  # [T, 2]


print('cbf_control_term defined.')\n@torch.no_grad()
def dpm_solver_1_cbf_sample(
    model,
    ve_diffusion,
    x_start,              # [B, 2]  start conditions
    x_goal,               # [B, 2]  goal  conditions
    obstacles,            # [N, 3]  float tensor (px, py, r) — normalised space
    T_steps=64,
    n_steps=25,
    k1=1.0,               # softmin temperature over obstacles  (m1 weights)
    k2=1.0,               # softmin temperature over waypoints  (m2 weights)
    alpha0=1.0,           # class-K gain
    gamma_delta=0.0,      # distance margin added to d_ij
    x_init=None,          # [B, T, 2] pre-sampled prior; if None, sample fresh
    device='cpu',
):
    """
    DPM-Solver-1 with trajectory-level CBF safety correction (eq. 70).

    At each denoising step:
        x_new = x - (sig_cur - sig_next) * eps_theta            # base ODE step
              - cbf_control_term(x, eps, ...)                   # safety correction

    The CBF correction is zero for the first half of denoising (rho = 0) and
    grows linearly for the second half, steering trajectories away from obstacles.

    Args:
        obstacles : [N, 3] tensor of (px, py, r).  Pass torch.zeros(0, 3) to
                    disable safety correction (recovers plain dpm_solver_1_sample).
        x_init    : optional [B, T, 2] tensor — shared prior for controlled comparison.
                    When provided, both CBF and no-CBF runs start from the same noise.

    Returns:
        x : [B, T_steps, 2]
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]

    x_start   = x_start.to(device)
    x_goal    = x_goal.to(device)
    obstacles = obstacles.to(device)

    sigmas  = ve_diffusion.sigmas.to(device)
    N_lvl   = ve_diffusion.n_levels
    indices   = torch.linspace(N_lvl, 0, n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]   # [n_steps+1], monotonically decreasing

    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    has_obstacles = obstacles.shape[0] > 0

    for step in range(n_steps):
        sig_cur  = sigma_seq[step]
        sig_next = sigma_seq[step + 1]

        eps_pred = model(x, sig_cur.expand(B), x_start, x_goal)   # [B, T, 2]

        # --- Base DPM-Solver-1 step -------------------------------------------
        x_new = x - (sig_cur - sig_next) * eps_pred

        # --- CBF correction (per sample in batch) -----------------------------
        if has_obstacles:
            for b in range(B):
                ctrl = cbf_control_term(
                    x[b], eps_pred[b],
                    sig_cur.item(), sig_next.item(), ve_diffusion.sigma_dot(sig_cur).item(),
                    step, n_steps,
                    obstacles, k1, k2, alpha0, gamma_delta,
                )
                x_new[b] = x_new[b] - ctrl   # eq. 70: subtract control term

        x = x_new

    return x   # [B, T_steps, 2]


# ---------------------------------------------------------------------------
# Demo: 4 contexts — DPM vs Safe DPM from the SAME prior
# Row 0: plain DPM-Solver-1 (no CBF)
# Row 1: Safe DPM-Solver-1  (CBF eq. 70)
# Same trajectory colour = same prior sample, different sampler
# ---------------------------------------------------------------------------
ema_model.eval()

OBSTACLES  = torch.tensor([[0.0, 0.0, 0.2]])   # one obstacle at centre, r=0.2
sigma_max  = ve.sigmas[ve.n_levels].item()      # prior scale = sigma at step 0

rng_cbf  = np.random.default_rng(55)
cbf_idxs = rng_cbf.choice(len(dataset.trajs), size=4, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for ax_i, tidx in enumerate(cbf_idxs):
    xs_i = dataset.starts[int(tidx)].to(device)
    xg_i = dataset.goals[int(tidx)].to(device)
    xs_b = xs_i.unsqueeze(0).expand(8, -1)
    xg_b = xg_i.unsqueeze(0).expand(8, -1)

    # 1. Sample shared prior
    x_init = torch.randn(8, T_STEPS, 2, device=device) * sigma_max

    # 2. Plain DPM (no CBF) — pass empty obstacles to short-circuit corrections
    samps_dpm = dpm_solver_1_cbf_sample(
        ema_model, ve, xs_b, xg_b,
        obstacles=torch.zeros(0, 3),
        T_steps=T_STEPS, n_steps=30,
        x_init=x_init, device=device,
    ).cpu().numpy()   # [8, T, 2]

    # 3. Safe DPM (with CBF) — same prior
    samps_safe = dpm_solver_1_cbf_sample(
        ema_model, ve, xs_b, xg_b,
        obstacles=OBSTACLES,
        T_steps=T_STEPS, n_steps=25,
        k1=1.0, k2=1.0, alpha0=1.0, gamma_delta=0.05,
        x_init=x_init, device=device,
    ).cpu().numpy()   # [8, T, 2]

    # Report min h value across safe batch
    obs_t = OBSTACLES.to(device)
    min_h = min(
        trajectory_cbf(
            torch.tensor(samps_safe[b], device=device),
            obs_t, k1=1.0, k2=1.0, gamma_delta=0.05,
        ).item()
        for b in range(8)
    )

    colors8 = plt.cm.tab10(np.linspace(0, 1, 8))

    for row, (samps, title) in enumerate([
        (samps_dpm,  f'DPM — Context {ax_i + 1}'),
        (samps_safe, f'Safe DPM — Context {ax_i + 1}  (min h={min_h:.3f})'),
    ]):
        ax = axes[row, ax_i]
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1],
                   s=0.2, alpha=0.06, c='gray', rasterized=True)
        for j in range(8):
            ax.plot(samps[j, :, 0], samps[j, :, 1],
                    '-', color=colors8[j], alpha=0.7, linewidth=1.0)
            ax.scatter(*samps[j, 0],  s=30, color=colors8[j], edgecolors='k', zorder=5, marker='s')
            ax.scatter(*samps[j, -1], s=40, color=colors8[j], edgecolors='k', zorder=5, marker='*')

        # Obstacle circle
        for px, py, r in OBSTACLES.numpy():
            ax.add_patch(plt.Circle((px, py), r, color='red', alpha=0.2,  zorder=4))
            ax.add_patch(plt.Circle((px, py), r, color='red', fill=False, linewidth=1.5, zorder=5))

        ax.scatter(*xs_i.cpu().numpy(), s=120, c='lime', edgecolors='black', zorder=6, marker='s')
        ax.scatter(*xg_i.cpu().numpy(), s=120, c='gold', edgecolors='black', zorder=6, marker='*')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)

fig.suptitle(
    f'Same prior — Row 1: DPM (no CBF)  |  Row 2: Safe DPM (CBF eq. 70) — {MPD_ENV}',
    fontsize=12, fontweight='bold',
)
plt.tight_layout()
plt.show()\n# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS_CBF  = 25
K1, K2       = 1.0, 1.0
ALPHA0       = 1.0
GAMMA_DELTA  = 0.0

OBSTACLES_DEMO = torch.tensor([[0.0, 0.0, 0.2]], device=device)  # (px, py, r)

# ── Pick one start/goal pair ──────────────────────────────────────────────────
rng_cbf = np.random.default_rng(55)
tidx    = int(rng_cbf.integers(0, len(dataset.trajs)))
xs_i    = dataset.starts[tidx].unsqueeze(0).to(device)   # [1, 2]
xg_i    = dataset.goals[tidx].unsqueeze(0).to(device)    # [1, 2]

# ── Build sigma schedule ──────────────────────────────────────────────────────
indices_cbf = torch.linspace(ve.n_levels, 0, N_STEPS_CBF + 1).long().clamp(0, ve.n_levels)
sigma_seq_cbf = ve.sigmas.to(device)[indices_cbf]         # [N_STEPS_CBF+1], decreasing

# ── Sample shared prior (both runs start from identical noise) ────────────────
torch.manual_seed(42)
x_prior = torch.randn(1, T_STEPS, 2, device=device) * sigma_seq_cbf[0]

# ── Run 1: Plain DPM-Solver-1 ─────────────────────────────────────────────────
ema_model.eval()
x_dpm = x_prior.clone()
with torch.no_grad():
    for step in range(N_STEPS_CBF):
        sig_cur  = sigma_seq_cbf[step]
        sig_next = sigma_seq_cbf[step + 1]
        eps_pred = ema_model(x_dpm, sig_cur.expand(1), xs_i, xg_i)
        x_dpm    = x_dpm - (sig_cur - sig_next) * eps_pred

# ── Run 2: Safe DPM-Solver-1, collecting per-step diagnostics ────────────────
x_cbf        = x_prior.clone()
h_before_log = []
omega_log    = []
h_after_log  = []

with torch.no_grad():
    for step in range(N_STEPS_CBF):
        sig_cur  = sigma_seq_cbf[step]
        sig_next = sigma_seq_cbf[step + 1]
        eps_pred = ema_model(x_cbf, sig_cur.expand(1), xs_i, xg_i)

        # h(X_t) BEFORE update
        h_before = trajectory_cbf(x_cbf[0], OBSTACLES_DEMO, K1, K2, GAMMA_DELTA).item()

        # base ODE step
        x_new = x_cbf - (sig_cur - sig_next) * eps_pred

        # CBF correction
        ctrl = cbf_control_term(
            x_cbf[0], eps_pred[0],
            # sig_cur.item(), sig_next.item(),
            indices_cbf[step].item(), indices_cbf[step + 1].item(), ve.sigma_dot(sig_cur).item(),
            step, N_STEPS_CBF,
            OBSTACLES_DEMO, K1, K2, ALPHA0, GAMMA_DELTA,
        )

        # ω — compute explicitly for logging (mirrors cbf_control_term logic)
        rho = max(0.0, step - N_STEPS_CBF / 2.0)
        if rho > 1e-8:
            grad_h    = grad_hXt_dXt(x_cbf[0], OBSTACLES_DEMO, K1, K2, GAMMA_DELTA)
            delta_sig = sig_cur.item() - sig_next.item()
            omega_val = (grad_h * (delta_sig * eps_pred[0])).sum().item() + ALPHA0 * h_before
        else:
            omega_val = float('nan')   # correction inactive in first half

        x_new[0] = x_new[0] - ctrl
        x_cbf    = x_new

        # h(X_t) AFTER update
        h_after = trajectory_cbf(x_cbf[0], OBSTACLES_DEMO, K1, K2, GAMMA_DELTA).item()

        h_before_log.append(h_before)
        omega_log.append(omega_val)
        h_after_log.append(h_after)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
steps_arr = np.arange(N_STEPS_CBF)

def _draw_traj(ax, traj_np, color, title):
    ax.scatter(bg_pts[:, 0], bg_pts[:, 1], s=0.2, alpha=0.06, c='gray', rasterized=True)
    ax.plot(traj_np[:, 0], traj_np[:, 1], '-', color=color, linewidth=1.8)
    ax.scatter(*traj_np[0],  s=60, color=color, edgecolors='k', zorder=5, marker='s')
    ax.scatter(*traj_np[-1], s=70, color=color, edgecolors='k', zorder=5, marker='*')
    for px, py, r in OBSTACLES_DEMO.cpu().numpy():
        ax.add_patch(plt.Circle((px, py), r, color='red', alpha=0.22, zorder=3))
        ax.add_patch(plt.Circle((px, py), r, color='red', fill=False, linewidth=1.5, zorder=4))
    ax.scatter(*xs_i[0].cpu().numpy(), s=120, c='lime', edgecolors='black', zorder=6, marker='s')
    ax.scatter(*xg_i[0].cpu().numpy(), s=120, c='gold', edgecolors='black', zorder=6, marker='*')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)

# Panel 1 — plain DPM
_draw_traj(axes[0], x_dpm[0].cpu().numpy(), 'steelblue', 'DPM-Solver-1 (no safety)')

# Panel 2 — safe DPM
h_final = h_after_log[-1]
_draw_traj(axes[1], x_cbf[0].cpu().numpy(), 'darkorange',
           f'Safe DPM-Solver-1  (h_final = {h_final:.3f})')

# Panel 3 — per-step diagnostics
ax3   = axes[2]
ax3_r = ax3.twinx()

ax3.plot(steps_arr, h_before_log, 'b-o',  markersize=3, linewidth=1.2, label='h(X_t) before')
ax3.plot(steps_arr, h_after_log,  'g-s',  markersize=3, linewidth=1.2, label='h(X_t) after')
ax3.axhline(0, color='red',  linestyle='--', linewidth=0.9, alpha=0.7, label='h = 0 (boundary)')
ax3.axvline(N_STEPS_CBF / 2, color='gray', linestyle=':', linewidth=1.0, label='ρ activates →')

valid    = ~np.isnan(omega_log)
omega_np = np.array(omega_log)
ax3_r.plot(steps_arr[valid], omega_np[valid], 'r-^',
           markersize=3, linewidth=1.0, alpha=0.8, label='ω')
ax3_r.axhline(0, color='tomato', linestyle=':', linewidth=0.8)
ax3_r.set_ylabel('ω', color='red', fontsize=10)
ax3_r.tick_params(axis='y', labelcolor='red')

ax3.set_xlabel('Denoising step')
ax3.set_ylabel('h(X_t)')
ax3.set_title('Per-step CBF diagnostics', fontsize=11)
ax3.legend(loc='upper left',  fontsize=8)
ax3_r.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.2)

fig.suptitle(f'Same prior — Safe vs Unsafe — {MPD_ENV}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()\nimport plotly.graph_objects as go
import plotly.express as px

# ---- Settings ----
N_ANIM_TRAJ   = 5
N_ANIM_STEPS  = 200
ANIM_STRIDE   = 4     # fewer frames = faster slider
PLOT3D_STRIDE = 4

# ---- Pick start / goal ----
rng7     = np.random.default_rng(42)
tidx7    = int(rng7.integers(0, len(dataset.trajs)))
x_start7 = dataset.starts[tidx7].to(device)
x_goal7  = dataset.goals[tidx7].to(device)

# ---- Run sampler ----
torch.manual_seed(0)
xs_b = x_start7.unsqueeze(0).expand(N_ANIM_TRAJ, -1)
xg_b = x_goal7.unsqueeze(0).expand(N_ANIM_TRAJ, -1)

ema_model.eval()
_, den_hist = ancestral_sample(
    ema_model, ve, xs_b, xg_b,
    T_steps=T_STEPS, n_steps=N_ANIM_STEPS, device=device,
    return_trajectory=True,
)
den_hist = den_hist.cpu().numpy()       # [steps+1, N_ANIM_TRAJ, T, 2]

frames_hist  = den_hist[::ANIM_STRIDE]
n_frames     = frames_hist.shape[0]

sigmas_np   = ve.sigmas.cpu().numpy()
indices_all = np.linspace(ve.n_levels, 0, N_ANIM_STEPS + 1).astype(int).clip(0, ve.n_levels)
sigma_frames = sigmas_np[indices_all][::ANIM_STRIDE]

s_np7 = x_start7.cpu().numpy()
g_np7 = x_goal7.cpu().numpy()

tab10_rgb = [
    f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
    for r, g, b, _ in [plt.cm.tab10(i / N_ANIM_TRAJ) for i in range(N_ANIM_TRAJ)]
]

# ============================================================
# Part A: Plotly animated scatter — 64 waypoints per frame
# ============================================================

def make_frame_traces(frame_idx):
    traces = []
    pts_frame = frames_hist[frame_idx]     # [N_ANIM_TRAJ, T, 2]
    for j in range(N_ANIM_TRAJ):
        pts = pts_frame[j]                 # [T, 2]
        traces.append(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode='markers',
            marker=dict(size=5, color=tab10_rgb[j]),
            name=f'traj {j+1}',
            legendgroup=f'traj{j}',
            showlegend=(frame_idx == 0),
        ))
    return traces

# Background data (subsample for speed)
bg_sub = bg_pts[::5]

# Build frames
plotly_frames = []
for fi in range(n_frames):
    plotly_frames.append(go.Frame(
        data=make_frame_traces(fi),
        name=str(fi),
        layout=go.Layout(title_text=f'frame {fi+1}/{n_frames}  |  σ={sigma_frames[fi]:.4f}'),
    ))

# Initial traces
init_traces = make_frame_traces(0)

# Background + start/goal (static, not part of animation frames)
init_traces += [
    go.Scatter(x=bg_sub[:,0], y=bg_sub[:,1], mode='markers',
               marker=dict(size=1, color='lightgray'), name='bg',
               showlegend=False, hoverinfo='skip'),
    go.Scatter(x=[s_np7[0]], y=[s_np7[1]], mode='markers',
               marker=dict(size=12, color='lime', symbol='square', line=dict(color='black',width=1)),
               name='start'),
    go.Scatter(x=[g_np7[0]], y=[g_np7[1]], mode='markers',
               marker=dict(size=14, color='gold', symbol='star', line=dict(color='black',width=1)),
               name='goal'),
]

fig_anim = go.Figure(
    data=init_traces,
    frames=plotly_frames,
    layout=go.Layout(
        title=f'Denoising animation  |  {MPD_ENV}',
        xaxis=dict(title='x', range=[-1.1, 1.1], scaleanchor='y'),
        yaxis=dict(title='y', range=[-1.1, 1.1]),
        width=620, height=620,
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.08, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(fi)],
                        dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                        label=f'{fi}')
                   for fi in range(n_frames)],
            transition=dict(duration=0),
            x=0, y=0, currentvalue=dict(prefix='frame: ', visible=True),
            len=1.0,
        )],
    ),
)
fig_anim.show()

# ============================================================
# Part B: Interactive 3D — frames stacked along z
# ============================================================
frames_3d = frames_hist[::PLOT3D_STRIDE]
n_3d      = frames_3d.shape[0]
z_vals    = np.linspace(0, 1, n_3d)

traces3d = []
for j in range(N_ANIM_TRAJ):
    for fi, z in enumerate(z_vals):
        pts = frames_3d[fi, j]
        traces3d.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1],
            z=np.full(T_STEPS, z),
            mode='markers',
            marker=dict(size=2, color=tab10_rgb[j]),
            opacity=0.12 + 0.88 * (fi / max(n_3d - 1, 1)),
            showlegend=(fi == n_3d - 1),
            name=f'traj {j+1}',
            legendgroup=f'traj{j}',
        ))

traces3d += [
    go.Scatter3d(x=[s_np7[0]], y=[s_np7[1]], z=[0.0], mode='markers',
                 marker=dict(size=6, color='lime', symbol='square'), name='start'),
    go.Scatter3d(x=[g_np7[0]], y=[g_np7[1]], z=[0.0], mode='markers',
                 marker=dict(size=8, color='gold', symbol='diamond'), name='goal'),
]

fig3d = go.Figure(data=traces3d)
fig3d.update_layout(
    title=dict(text=f'Denoising frames stacked in z  |  {MPD_ENV}', x=0.5),
    scene=dict(
        xaxis=dict(title='x', range=[-1.1, 1.1]),
        yaxis=dict(title='y', range=[-1.1, 1.1]),
        zaxis=dict(title='denoising step  (0=noisy → 1=clean)'),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1.5),
    ),
    width=800, height=700,
    legend=dict(x=1.02, y=0.5),
)
fig3d.show()