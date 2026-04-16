"""
SafeDPMSolver CFG Visualiser — FastAPI Backend
===============================================
Uses Classifier-Free Guidance (CFG) with rasterized occupancy-map conditioning.

Plain DPM sampling uses CFG with rasterized-map conditioning:
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

Safe DPM sampling uses CFG as the base ODE, then applies the CBF safety
filter using the raw ellipsoid geometry.

Run with:
    conda run -n py_3_10 uvicorn EllipsoidalCBFSampling.visualizer_app_cfg:app \
        --host 0.0.0.0 --port 8002 --reload \
        --reload-dir /home/earth/SDPMSP_Clone/EllipsoidalCBFSampling

Endpoints:
    GET  /                   -> redirect to /static/index.html
    GET  /api/models         -> list available .pt checkpoints
    POST /api/run            -> run CFG Plain + CFG+CBF Safe from identical prior
    POST /api/batch_run      -> run N random-start/goal samples in one GPU batch
    POST /api/math           -> re-evaluate CBF metrics for a given trajectory
    POST /api/recompute_ctrl -> recompute CBF + control for a modified trajectory
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — add project root so sibling packages are importable
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

MAZE_DATA_BASE = PROJECT_DIR.parent / 'Diffuser' / 'data'
MAZE_ENVS: dict[str, Path] = {
    'PointMaze_UMaze-v3': MAZE_DATA_BASE / 'umaze_v2',
    'PointMaze_Large-v3': MAZE_DATA_BASE / 'large_v2',
}

from EllipsoidalCBFSampling.models.score_net_ellipsoids import TemporalUnet
from EllipsoidalCBFSampling.models.ve_diffusion_ellipsoids import VEDiffusion
from EllipsoidalCBFSampling.models.samplers_ellipsoids_cfg import (
    dpm_solver_1_cfg_sample,
    dpm_solver_1_cbf_cfg_sample,
    recompute_cbf_step,
)
from EllipsoidalCBFSampling.CBF.trajectory_cbf_ellipses import compute_cbf_metrics
from EllipsoidalCBFSampling.rasterize import get_large_maze_wall_bitmap, rasterize_scene

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default start / goal used when the caller does not specify them
DEFAULT_START = [-0.8, -0.8]
DEFAULT_GOAL  = [0.8,  0.8]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title='SafeDPMSolver CFG Visualiser', version='2.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(PROJECT_DIR / 'static_cfg')), name='static')


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/static/index.html')


# ---------------------------------------------------------------------------
# Model cache  (loaded lazily, kept in memory)
# Cache entry: (ema_model, ve, T_steps, wall_bitmap, map_h, map_w)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _load_model(model_name: str):
    """Load and cache (ema_model, ve, T_steps, wall_bitmap, map_h, map_w).

    Expects the rasterized checkpoint format saved by train_large_ellipsoids_diffuser.ipynb,
    with a 'config' key containing local_dim, global_dim, map_h, map_w, xy_min, xy_max.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_path = CHECKPOINTS_DIR / model_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    if 'config' not in ckpt:
        raise KeyError(
            f'Unrecognised checkpoint format — expected "config" key, got: {list(ckpt.keys())}'
        )

    cfg     = ckpt['config']
    T_steps = cfg['T_steps']
    map_h   = cfg.get('map_h', 64)
    map_w   = cfg.get('map_w', 64)

    score_net = TemporalUnet(
        state_dim=2,
        T_steps=T_steps,
        unet_input_dim=cfg.get('unet_input_dim', 32),
        dim_mults=tuple(cfg.get('dim_mults', [1, 2, 4])),
        local_dim=cfg.get('local_dim', 16),
        global_dim=cfg.get('global_dim', 256),
    ).to(DEVICE)

    ema_model = copy.deepcopy(score_net).to(DEVICE)
    ema_model.load_state_dict(ckpt['score_net'])

    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    ve = VEDiffusion(
        model=ema_model,
        sigma_min=cfg.get('sigma_min', 0.01),
        sigma_max=cfg.get('sigma_max', 10.0),
        n_levels=cfg.get('n_levels', 1000),
    ).to(DEVICE)

    # Build and cache the wall bitmap using the normalisation stored in the checkpoint
    xy_min   = np.array(cfg['xy_min'],  dtype=np.float32)
    xy_max   = np.array(cfg['xy_max'],  dtype=np.float32)
    xy_range = xy_max - xy_min
    wall_bitmap = get_large_maze_wall_bitmap(xy_min, xy_range, map_h, map_w).to(DEVICE)

    _model_cache[model_name] = (ema_model, ve, T_steps, wall_bitmap, map_h, map_w)
    return ema_model, ve, T_steps, wall_bitmap, map_h, map_w


# ---------------------------------------------------------------------------
# Helper: rasterize API obstacles → [B, 1, H, W] occupancy map
#
# Obstacles from the frontend are already in normalised [-1, 1] space
# (same coordinate system as start/goal), so no coordinate conversion is
# needed — pass them directly to rasterize_scene.
# ---------------------------------------------------------------------------

def _build_occ_map(
    obstacles:   list,
    wall_bitmap: torch.Tensor,   # [1, 1, H, W]
    map_h:       int,
    map_w:       int,
    B:           int = 1,
) -> torch.Tensor:
    """Rasterize API obstacles (normalised space) + walls → [B, 1, H, W]."""
    data = [[obs.x, obs.y, obs.a, obs.b] for obs in obstacles]
    if not data:
        data = [[0.0, 0.0, 0.0, 0.0]]   # no obstacles → blank map (walls only)
    ells = torch.tensor(data, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, N, 4]
    occ = rasterize_scene(ells, wall_bitmap, map_h, map_w)   # [1, 1, H, W]
    if B > 1:
        occ = occ.expand(B, -1, -1, -1).contiguous()         # [B, 1, H, W]
    return occ


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

@app.get('/api/models')
def list_models():
    """Return sorted list of .pt filenames in EllipsoidalCBFSampling/checkpoints/."""
    if not CHECKPOINTS_DIR.exists():
        return {'models': []}
    models = sorted(p.name for p in CHECKPOINTS_DIR.glob('*.pt'))
    return {'models': models}


# ---------------------------------------------------------------------------
# /api/envs  — list maze environments that have data on disk
# ---------------------------------------------------------------------------

@app.get('/api/envs')
def list_envs():
    """Return list of env_ids for which maze data exists on disk."""
    return {'envs': [e for e, d in MAZE_ENVS.items() if d.exists()]}


# ---------------------------------------------------------------------------
# /api/maze  — wall geometry in normalised coordinates
# ---------------------------------------------------------------------------

def _build_maze_data(env_id: str) -> dict:
    """Load maze geometry for env_id and return walls, view bounds, and metadata."""
    if env_id not in MAZE_ENVS:
        raise HTTPException(status_code=400, detail=f'Unknown env_id: {env_id!r}. Available: {list(MAZE_ENVS)}')
    maze_data_dir = MAZE_ENVS[env_id]
    if not maze_data_dir.exists():
        raise HTTPException(status_code=404, detail=f'Maze data not found: {maze_data_dir}')

    meta      = json.loads((maze_data_dir / 'metadata.json').read_text())
    maze_map  = meta['maze_map']
    cell_size = float(meta['cell_size'])
    d_min     = meta['norm']['d_min']
    d_max     = meta['norm']['d_max']
    d_range   = [d_max[0] - d_min[0], d_max[1] - d_min[1]]
    n_rows    = len(maze_map)
    n_cols    = len(maze_map[0])
    half_w    = cell_size / d_range[0]
    half_h    = cell_size / d_range[1]

    walls = []
    for r in range(n_rows):
        for c in range(n_cols):
            if maze_map[r][c] == 1:
                wx = (c - (n_cols - 1) / 2.0) * cell_size
                wy = ((n_rows - 1) / 2.0 - r)  * cell_size
                nx = 2.0 * (wx - d_min[0]) / d_range[0] - 1.0
                ny = 2.0 * (wy - d_min[1]) / d_range[1] - 1.0
                walls.append({'cx': nx, 'cy': ny, 'hw': half_w, 'hh': half_h})

    padding  = max(half_w, half_h) * 2
    view_min = round(-(1.0 + padding), 3)
    view_max = round(  1.0 + padding,  3)

    return {
        'walls':    walls,
        'view_min': view_min,
        'view_max': view_max,
        'maze_map': maze_map,
        'd_min':    d_min,
        'd_max':    d_max,
    }


@app.get('/api/maze')
def get_maze(env_id: str = 'PointMaze_UMaze-v3'):
    """Return maze wall rectangles in normalised [-1, 1] coordinate space."""
    return _build_maze_data(env_id)


# ---------------------------------------------------------------------------
# /api/run   — run CFG Plain + CFG+CBF Safe from identical prior
# ---------------------------------------------------------------------------

class Obstacle(BaseModel):
    x: float = 0.0
    y: float = 0.0
    a: float = 0.3
    b: float = 0.3

class RunRequest(BaseModel):
    model_name:     str   = 've_unet_largemaze_rasterized_v1.pt'
    n_steps:        int   = 20
    c:              float = 1.0
    k1:             float = 1.0
    k2:             float = 1.0
    gamma_delta:    float = 0.0
    obstacles:      List[Obstacle] = [Obstacle()]
    start_x:        float = -0.8
    start_y:        float = -0.8
    goal_x:         float = 0.8
    goal_y:         float = 0.8
    alpha0:         float = 1.0
    use_softplus:   bool  = True
    guidance_scale: float = 3.0
    seed:           Optional[int] = None


@app.post('/api/run')
def run_optimisation(req: RunRequest):
    """
    Run both CFG Plain DPM-Solver-1 and CFG+CBF Safe DPM-Solver-1 from the
    **exact same** initial noise prior.  Returns per-step trajectory history
    for both runs, plus per-step CBF metrics for the safe trajectory.

    The same UI-drawn obstacles serve two roles:
      - occ_map  [1, 1, H, W] — rasterized (walls + ellipsoids) fed to the score
        network as the scene conditioning signal.
      - obstacles [N, 4]      — raw geometry for CBF safety math (cx,cy,a,b in
        normalised space, matching what was passed to the visualiser).

    Response shape:
        n_steps          : int
        T_steps          : int
        start            : [2]
        goal             : [2]
        prior            : [T, 2]
        plain_history    : [n_steps+1, T, 2]
        before_history   : [n_steps+1, T, 2]
        safe_history     : [n_steps+1, T, 2]
        cbf_step_data    : [n_steps+1] dicts
    """
    try:
        ema_model, ve, T_steps, wall_bitmap, map_h, map_w = _load_model(req.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Model load error: {exc}')

    k2 = max(req.k2, 1e-3)

    # occ_map [1, 1, H, W] — rasterized scene for model conditioning
    occ_map = _build_occ_map(req.obstacles, wall_bitmap, map_h, map_w, B=1)

    # obstacles [N, 4] — raw geometry for CBF safety math
    obs_list  = [[obs.x, obs.y, obs.a, obs.b] for obs in req.obstacles]
    obstacles = torch.tensor(obs_list, dtype=torch.float32, device=DEVICE)

    x_start = torch.tensor([req.start_x, req.start_y], dtype=torch.float32, device=DEVICE)
    x_goal  = torch.tensor([req.goal_x,  req.goal_y],  dtype=torch.float32, device=DEVICE)

    # Sample shared prior
    if req.seed is not None:
        torch.manual_seed(req.seed)
    sigmas    = ve.sigmas.to(DEVICE)
    N_lvl     = ve.n_levels
    indices   = torch.linspace(N_lvl, 0, req.n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]
    x_init    = torch.randn(1, T_steps, 2, device=DEVICE) * sigma_seq[0]

    import time

    try:
        # --- CFG Plain DPM-Solver-1 ---
        t0 = time.perf_counter()
        _, plain_history = dpm_solver_1_cfg_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            occ_map=occ_map,
            T_steps=T_steps, n_steps=req.n_steps,
            guidance_scale=req.guidance_scale,
            device=DEVICE, x_init=x_init.clone(),
            return_history=True,
        )
        plain_time = time.perf_counter() - t0

        # --- CFG + CBF Safe DPM-Solver-1 ---
        t0 = time.perf_counter()
        _, safe_history, before_history, cbf_history = dpm_solver_1_cbf_cfg_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            occ_map=occ_map,
            obstacles=obstacles,
            T_steps=T_steps, n_steps=req.n_steps,
            guidance_scale=req.guidance_scale,
            k1=req.k1, k2=k2, c=req.c,
            alpha0=req.alpha0, gamma_delta=req.gamma_delta,
            use_softplus=req.use_softplus,
            device=DEVICE, x_init=x_init.clone(),
            return_history=True,
        )
        safe_time = time.perf_counter() - t0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Sampler error: {exc}')

    def traj_to_list(t):
        return t[0].cpu().tolist()

    return {
        'n_steps':        req.n_steps,
        'T_steps':        T_steps,
        'start':          [req.start_x, req.start_y],
        'goal':           [req.goal_x,  req.goal_y],
        'prior':          traj_to_list(x_init),
        'plain_history':  [traj_to_list(h) for h in plain_history],
        'before_history': [traj_to_list(h) for h in before_history],
        'safe_history':   [traj_to_list(h) for h in safe_history],
        'cbf_step_data':  cbf_history,
        'plain_time':     round(plain_time, 3),
        'safe_time':      round(safe_time,  3),
    }


# ---------------------------------------------------------------------------
# /api/batch_run  — run N samples (random starts/goals) in one GPU batch
# ---------------------------------------------------------------------------

class BatchRunRequest(BaseModel):
    model_name:     str   = 've_unet_largemaze_rasterized_v1.pt'
    n_steps:        int   = 20
    n_samples:      int   = 100
    c:              float = 1.0
    k1:             float = 1.0
    k2:             float = 1.0
    gamma_delta:    float = 0.0
    obstacles:      List[Obstacle] = [Obstacle()]
    alpha0:         float = 1.0
    use_softplus:   bool  = True
    guidance_scale: float = 3.0
    seed:           Optional[int] = None


@app.post('/api/batch_run')
def batch_run(req: BatchRunRequest):
    """
    Run N independent trajectories (random starts/goals) in a single batched
    GPU forward pass.  Returns only final trajectories (no step history).

    occ_map  [N, 1, H, W] — same rasterized scene broadcast across all N samples.
    obstacles [M, 4]       — raw geometry for CBF safety filter.
    """
    try:
        ema_model, ve, T_steps, wall_bitmap, map_h, map_w = _load_model(req.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Model load error: {exc}')

    k2 = max(req.k2, 1e-3)
    N  = req.n_samples

    if req.seed is not None:
        torch.manual_seed(req.seed)

    # occ_map [N, 1, H, W] — same scene for all N samples
    occ_map = _build_occ_map(req.obstacles, wall_bitmap, map_h, map_w, B=N)

    # obstacles [M, 4] — raw geometry for CBF
    obs_list  = [[obs.x, obs.y, obs.a, obs.b] for obs in req.obstacles]
    obstacles = torch.tensor(obs_list, dtype=torch.float32, device=DEVICE)

    # Random starts/goals uniformly in [-0.9, 0.9]²
    starts = (torch.rand(N, 2, device=DEVICE) * 2 - 1) * 0.9
    goals  = (torch.rand(N, 2, device=DEVICE) * 2 - 1) * 0.9

    # Build batched prior [N, T_steps, 2]
    sigmas    = ve.sigmas.to(DEVICE)
    N_lvl     = ve.n_levels
    indices   = torch.linspace(N_lvl, 0, req.n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]
    x_init    = torch.randn(N, T_steps, 2, device=DEVICE) * sigma_seq[0]

    import time
    try:
        t0 = time.perf_counter()
        plain = dpm_solver_1_cfg_sample(
            ema_model, ve,
            x_start=starts, x_goal=goals,
            occ_map=occ_map,
            T_steps=T_steps, n_steps=req.n_steps,
            guidance_scale=req.guidance_scale,
            device=DEVICE, x_init=x_init.clone(),
            return_history=False,
        )
        plain_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        safe = dpm_solver_1_cbf_cfg_sample(
            ema_model, ve,
            x_start=starts, x_goal=goals,
            occ_map=occ_map,
            obstacles=obstacles,
            T_steps=T_steps, n_steps=req.n_steps,
            guidance_scale=req.guidance_scale,
            k1=req.k1, k2=k2, c=req.c,
            alpha0=req.alpha0, gamma_delta=req.gamma_delta,
            use_softplus=req.use_softplus,
            device=DEVICE, x_init=x_init.clone(),
            return_history=False,
        )
        safe_time = time.perf_counter() - t0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Batch sampler error: {exc}')

    return {
        'plain_trajs': plain.cpu().tolist(),
        'safe_trajs':  safe.cpu().tolist(),
        'starts':      starts.cpu().tolist(),
        'goals':       goals.cpu().tolist(),
        'n_samples':   N,
        'T_steps':     T_steps,
        'n_steps':     req.n_steps,
        'obstacles':   [[obs.x, obs.y, obs.a, obs.b] for obs in req.obstacles],
        'gamma_delta': req.gamma_delta,
        'plain_time':  round(plain_time, 3),
        'safe_time':   round(safe_time,  3),
        'mode':        'rasterized',
    }


# ---------------------------------------------------------------------------
# /api/math   — live re-evaluation for a given traj + parameters
# ---------------------------------------------------------------------------

class MathRequest(BaseModel):
    traj:         List[List[float]]   # [T, 2]
    c:            float = 1.0
    k1:           float = 1.0
    k2:           float = 1.0
    gamma_delta:  float = 0.0
    obstacles:    List[Obstacle] = [Obstacle()]
    use_softplus: bool = True


@app.post('/api/math')
def evaluate_math(req: MathRequest):
    """
    Re-evaluate CBF metrics for a user-provided trajectory and parameters.
    Called on parameter-slider changes or waypoint drags (no re-sampling).
    """
    k2 = max(req.k2, 1e-3)

    Xt = torch.tensor(req.traj, dtype=torch.float32, device=DEVICE)
    obs_list  = [[obs.x, obs.y, obs.a, obs.b] for obs in req.obstacles]
    obstacles = torch.tensor(obs_list, dtype=torch.float32, device=DEVICE)

    try:
        metrics = compute_cbf_metrics(Xt, obstacles, req.c, req.k1, k2, req.gamma_delta, req.use_softplus)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'CBF math error: {exc}')

    return metrics


# ---------------------------------------------------------------------------
# /api/recompute_ctrl  — recompute CBF metrics + control + after trajectory
#                        for a (possibly user-modified) before-control trajectory
# ---------------------------------------------------------------------------

class RecomputeRequest(BaseModel):
    before_traj:  List[List[float]]   # [T, 2] — possibly user-modified
    eps_pred_x:   List[float]         # [T]    — cached from original run
    eps_pred_y:   List[float]         # [T]
    sigma_delta:  float
    sigma_dot:    float
    noise_idx:    int
    step_delta:   int   = 0
    n_steps:      int
    c:            float = 1.0
    k1:           float = 1.0
    k2:           float = 1.0
    gamma_delta:  float = 0.0
    obstacles:    List[Obstacle] = [Obstacle()]
    alpha0:       float = 1.0
    use_softplus: bool  = True


@app.post('/api/recompute_ctrl')
def recompute_ctrl_endpoint(req: RecomputeRequest):
    """
    Recompute CBF metrics and after-control trajectory for one step,
    given a (possibly modified) before-control trajectory and cached eps_pred.
    Does not re-run the score network.
    """
    k2 = max(req.k2, 1e-3)

    obs_list  = [[obs.x, obs.y, obs.a, obs.b] for obs in req.obstacles]
    obstacles = torch.tensor(obs_list, dtype=torch.float32, device=DEVICE)

    try:
        result = recompute_cbf_step(
            before_traj  = req.before_traj,
            eps_pred_x   = req.eps_pred_x,
            eps_pred_y   = req.eps_pred_y,
            sigma_delta  = req.sigma_delta,
            sigma_dot    = req.sigma_dot,
            noise_idx    = req.noise_idx,
            step_delta   = req.step_delta,
            n_steps      = req.n_steps,
            obstacles    = obstacles,
            k1           = req.k1,
            k2           = k2,
            c            = req.c,
            alpha0       = req.alpha0,
            gamma_delta  = req.gamma_delta,
            use_softplus = req.use_softplus,
            device       = DEVICE,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Recompute error: {exc}')

    return result
