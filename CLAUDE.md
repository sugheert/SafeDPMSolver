# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafeDPMSolverProject implements **safe trajectory generation** using Variance-Exploding (VE) Score-Based Diffusion Models combined with **Control Barrier Functions (CBFs)** for obstacle avoidance. It is designed for 2D motion planning using neural network-based diffusion.

The project is imported as `SafeDPMSolver` (not `SafeDPMSolverProject`). The parent directory of `SafeDPMSolverProject/` must be on `sys.path`.

## Environment

Use conda environment `py_3_10`:
```
conda activate py_3_10
```

## External Dependencies

- **torch_robotics** (MPD): Located at `C:\Users\Owner\SAFE_DIFFUSION\mpd-public\deps` — loaded dynamically in `environments/circles_obstacles.py`
- **MPD training data**: `C:\Users\Owner\Downloads\MPD\mpd-public\data_trajectories` — `.pt` files with shape `[n_trajs, 64, 4]` (x, y, vx, vy)
- Python packages: `torch`, `einops`, `numpy`, `matplotlib`

## Development Workflow

This project is **notebook-driven**. Primary development and experimentation happen in:
- `notebooks/train_and_sample.ipynb` — training/sampling on MPD environments (e.g., `EnvDense2D-RobotPointMass`)
- `notebooks/train_and_sample_circles.ipynb` — training/sampling on circles-only environments
- `notebooks/train_and_sample_circles_copy.ipynb` — training/sampling on circles-only environments: the latest version

Run notebooks from the `notebooks/` directory; they use `os.getcwd()` to construct paths to checkpoints and project root.

There are no build scripts, test suites, or CLI entry points.

## Architecture

### Data Flow

```
MPD .pt files → MPDTrajectoryDataset → [B, T, 2] trajectories
    → VEDiffusion (add noise) → ScoreNet (predict noise) → DSM loss
    → ancestral sampler (Euler-Maruyama) → clean trajectories
    → CBF constraint check/gradient for safe sampling
```

### Key modules

**`models/score_net.py`** — `TemporalUnet` (aliased as `ScoreNet`)
- 1D temporal U-Net for denoising trajectories of shape `[B, T, 2]`
- Conditioning: concatenates sinusoidal sigma embedding + start/goal positions
- Architecture: `ResidualTemporalBlock` with `Conv1dBlock` (Conv1d → GroupNorm → Mish), down/up sample paths

**`models/ve_diffusion.py`** — `VEDiffusion`
- VE-SDE wrapper with geometric noise schedule: `σ_i = σ_min * (σ_max/σ_min)^(i/N)`
- Computes denoising score matching loss: `‖ε_θ(x₀ + σε, σ, x_start, x_goal) − ε‖²`

**`CBF/trajectory_cbf.py`** — trajectory-level CBF
- Two-level softmin structure: per-obstacle → per-waypoint → per-trajectory
- Provides `trajectory_cbf()` for safety value and `grad_hXt_dXt()` for gradient `[T, 2]`
- Key parameters: `k1` (obstacle softmin temperature), `k2` (waypoint softmin temperature), `gamma_delta` (safety margin)

**`environments/circles_obstacles.py`** — `get_circles_from_env()`
- Loads MPD environments dynamically and extracts `Circle(x, y, r)` obstacle lists

### Checkpoint Format

Checkpoints saved via `torch.save` contain keys: `step`, `score_net`, `ema_model`, `optimizer`, `loss_history`, `model_cfg` (for `TemporalUnet`), `ve_cfg` (for `VEDiffusion`). Reconstruct models from saved `model_cfg`/`ve_cfg` before loading state dicts.

### Training Setup (default hyperparameters)

| Parameter | Value |
|-----------|-------|
| `T_STEPS` | 64 waypoints |
| `UNET_INPUT_DIM` | 32 |
| `DIM_MULTS` | (1, 2, 4) |
| `SIGMA_MIN / SIGMA_MAX` | 0.01 / 10.0 |
| `N_LEVELS` | 1000 |
| `BATCH_SIZE` | 128 |
| `LR` | 3e-4 |
| `EMA_DECAY` | 0.995 |

Sampling uses an Euler-Maruyama ancestral sampler (`ancestral_sample()`), denoising from pure noise over `n_steps` (default 200).
