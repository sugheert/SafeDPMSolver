"""Smoke test for train_umaze_diffuser notebook cells."""
import sys, os, copy
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # notebooks/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)               # SafeDPMSolverProject/
PARENT       = os.path.dirname(PROJECT_ROOT)               # parent dir (SafeDPMSolver importable)

for p in [PARENT, PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'Diffuser')]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("=== 1. Imports ===")
from SafeDPMSolver.models.score_net    import TemporalUnet    # from C:\Users\Owner\SafeDPMSolver\
from SafeDPMSolver.models.ve_diffusion import VEDiffusion
from models.samplers                   import dpm_solver_1_sample   # from SafeDPMSolverProject\models\
from maze_dataset                      import MazeDataset, render_maze_ax  # from SafeDPMSolverProject\Diffuser\
print("OK")

# ── 2. Dataset ────────────────────────────────────────────────────────────────
print("\n=== 2. Dataset ===")
DATA_DIR = os.path.join(PROJECT_ROOT, 'Diffuser', 'data', 'umaze_v2')
ds = MazeDataset(DATA_DIR, xy_only=True)
print(ds)
sample0 = ds[0]
print("Single item shape:", sample0.shape)

from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
batch = next(iter(dl))
print("Batch shape:", batch.shape)
xs = batch[:, 0, :]
xg = batch[:, -1, :]
print("xs:", xs.shape, "  xg:", xg.shape)

# ── 3. Model ──────────────────────────────────────────────────────────────────
print("\n=== 3. Model ===")
device = 'cpu'
T_STEPS = ds.horizon
score_net = TemporalUnet(
    state_dim=2, T_steps=T_STEPS, unet_input_dim=32, dim_mults=(1, 2, 4)
).to(device)
ve = VEDiffusion(score_net, 0.01, 10.0, 1000).to(device)
ema_model = copy.deepcopy(score_net).to(device)
for p in ema_model.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.Adam(score_net.parameters(), lr=3e-4)
print(f"TemporalUnet params: {sum(p.numel() for p in score_net.parameters()):,}")

# ── 4. Training step ──────────────────────────────────────────────────────────
print("\n=== 4. Training step ===")
score_net.train()
batch = batch.to(device)
xs, xg = batch[:, 0, :], batch[:, -1, :]
loss, info = ve(batch, xs, xg)
print(f"Loss: {loss.item():.4f}  |  sigma_mean: {info['sigma_mean']:.4f}")
assert torch.isfinite(loss), "Loss is not finite!"
loss.backward()
optimizer.step()
print("Backward + optimizer step OK")

# EMA update
EMA_DECAY = 0.995
with torch.no_grad():
    for p_ema, p in zip(ema_model.parameters(), score_net.parameters()):
        p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)
print("EMA update OK")

# ── 5. DPM-Solver-1 sample ────────────────────────────────────────────────────
print("\n=== 5. DPM-Solver-1 sample ===")
ema_model.eval()
samp = dpm_solver_1_sample(
    ema_model, ve,
    x_start=xs[:1], x_goal=xg[:1],
    T_steps=T_STEPS, n_steps=5, device=device,
)
print("Sample shape:", samp.shape)
assert samp.shape == (1, T_STEPS, 2), f"Expected (1, {T_STEPS}, 2), got {samp.shape}"
world = ds.unnormalize(samp.cpu())
print(f"World x range: [{world[0,:,0].min():.3f}, {world[0,:,0].max():.3f}]")
print(f"World y range: [{world[0,:,1].min():.3f}, {world[0,:,1].max():.3f}]")

# ── 6. Unnormalize xs/xg (as used in notebook for scatter) ───────────────────
print("\n=== 6. Unnormalize xs/xg ===")
xs_w = ds.unnormalize(xs[:1].cpu())
xg_w = ds.unnormalize(xg[:1].cpu())
print("xs_w shape:", xs_w.shape)   # expect [1, 2]
print("xg_w shape:", xg_w.shape)

# ── 7. render_maze_ax ────────────────────────────────────────────────────────
print("\n=== 7. render_maze_ax ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
render_maze_ax(ax, ds.maze_map, ds.cell_size)
ax.plot(world[0, :, 0], world[0, :, 1], color='steelblue')
ax.scatter(xs_w[0, 0], xs_w[0, 1], c='green', s=60, zorder=5)
ax.scatter(xg_w[0, 0], xg_w[0, 1], c='red',   s=60, zorder=5)
plt.savefig(os.path.join(os.path.dirname(__file__), 'smoke_test_output.png'))
plt.close()
print("Plot saved → smoke_test_output.png")

# ── 8. Checkpoint round-trip ─────────────────────────────────────────────────
print("\n=== 8. Checkpoint save/load ===")
import tempfile
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    ckpt_path = f.name
torch.save({
    'step': 1,
    'score_net': score_net.state_dict(),
    'ema_model': ema_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss_history': [loss.item()],
    'config': {
        'T_steps': T_STEPS, 'unet_input_dim': 32, 'dim_mults': [1, 2, 4],
        'sigma_min': 0.01, 'sigma_max': 10.0, 'n_levels': 1000,
        'dataset': DATA_DIR,
    },
}, ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg  = ckpt['config']
net2 = TemporalUnet(state_dim=2, T_steps=cfg['T_steps'],
                    unet_input_dim=cfg['unet_input_dim'],
                    dim_mults=tuple(cfg['dim_mults'])).to(device)
net2.load_state_dict(ckpt['score_net'])
os.unlink(ckpt_path)
print("Checkpoint round-trip OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
