"""
SafeDPMSolver UMaze Interactive Visualiser — FastAPI Backend
============================================================
Run with:
    conda run -n py_3_10 uvicorn Diffuser.visualization.visualizer_diffuser_style_app:app --host 0.0.0.0 --port 8002 --reload --reload-dir "c:/Users/Owner/SafeDPMSolverProject" --app-dir "c:/Users/Owner/SafeDPMSolverProject"

Endpoints:
    GET  /                  -> redirect to /static/index.html
    GET  /api/models        -> list available .pt checkpoints
    GET  /api/maze          -> maze wall rectangles in normalised coordinates
    POST /api/run           -> run Plain + Safe DPM from identical prior, return full step history
    POST /api/math          -> re-evaluate CBF metrics for a given trajectory + params
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — add project root so sibling packages are importable
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(__file__).parent.parent.parent.resolve()   # SafeDPMSolverProject/
STATIC_DIR    = Path(__file__).parent / 'static'
MAZE_DATA_DIR = PROJECT_DIR / 'Diffuser' / 'data' / 'umaze_v2'

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.score_net import TemporalUnet
from models.ve_diffusion import VEDiffusion
from models.samplers import dpm_solver_1_sample, dpm_solver_1_cbf_sample, recompute_cbf_step
from CBF.trajectory_cbf import compute_cbf_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_START = [-0.8, -0.8]
DEFAULT_GOAL  = [0.8,  0.8]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title='SafeDPMSolver UMaze Visualiser', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/static/index.html')


# ---------------------------------------------------------------------------
# Model cache  (loaded lazily, kept in memory)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _load_model(model_name: str):
    """Load and cache (ema_model, ve_diffusion, T_steps) for a given checkpoint name.

    Supports both the original format (model_cfg / ve_cfg keys) and the
    diffuser-style format where everything is packed into a single 'config' key.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_path = CHECKPOINTS_DIR / model_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)

    # ── Reconstruct model_cfg / ve_cfg from whichever format is present ──
    if 'model_cfg' in ckpt and 've_cfg' in ckpt:
        # Original circles-checkpoint format
        model_cfg = ckpt['model_cfg']
        ve_cfg    = ckpt['ve_cfg']
        T_steps   = model_cfg.get('T_steps', 64)
    elif 'config' in ckpt:
        # Diffuser-style checkpoint (umaze)
        cfg = ckpt['config']
        T_steps   = cfg['T_steps']
        model_cfg = {
            'state_dim':      2,
            'T_steps':        T_steps,
            'unet_input_dim': cfg.get('unet_input_dim', 32),
            'dim_mults':      tuple(cfg.get('dim_mults', [1, 2, 4])),
        }
        ve_cfg = {
            'sigma_min': cfg.get('sigma_min', 0.01),
            'sigma_max': cfg.get('sigma_max', 10.0),
            'n_levels':  cfg.get('n_levels',  1000),
        }
    else:
        raise KeyError(f'Unrecognised checkpoint format — keys: {list(ckpt.keys())}')

    score_net = TemporalUnet(**model_cfg).to(DEVICE)
    ema_model = copy.deepcopy(score_net).to(DEVICE)
    ema_model.load_state_dict(ckpt['ema_model'])
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    ve = VEDiffusion(model=ema_model, **ve_cfg).to(DEVICE)

    _model_cache[model_name] = (ema_model, ve, T_steps)
    return ema_model, ve, T_steps


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

@app.get('/api/models')
def list_models():
    """Return sorted list of .pt filenames in the checkpoints/ directory."""
    if not CHECKPOINTS_DIR.exists():
        return {'models': []}
    models = sorted(p.name for p in CHECKPOINTS_DIR.glob('*.pt'))
    return {'models': models}


# ---------------------------------------------------------------------------
# /api/maze  — wall geometry in normalised coordinates
# ---------------------------------------------------------------------------

@app.get('/api/maze')
def get_maze():
    """
    Return U-Maze wall rectangles pre-computed in the same normalised [-1, 1]
    coordinate space used by the diffusion model.

    Response shape:
        walls     : list of {cx, cy, hw, hh}  — centre + half-extents (normalised)
        view_min  : float   — recommended canvas lower bound
        view_max  : float   — recommended canvas upper bound
        maze_map  : [[int]] — raw 0/1 grid (1 = wall)
        d_min     : [2]     — normalisation lower bound (world coords)
        d_max     : [2]     — normalisation upper bound (world coords)
    """
    if not MAZE_DATA_DIR.exists():
        raise HTTPException(status_code=404, detail=f'Maze data not found: {MAZE_DATA_DIR}')

    meta      = json.loads((MAZE_DATA_DIR / 'metadata.json').read_text())
    maze_map  = meta['maze_map']
    cell_size = float(meta['cell_size'])
    d_min     = meta['norm']['d_min']   # [x_min, y_min]
    d_max     = meta['norm']['d_max']   # [x_max, y_max]
    d_range   = [d_max[0] - d_min[0], d_max[1] - d_min[1]]

    n_rows = len(maze_map)
    n_cols = len(maze_map[0])

    def to_norm(wx: float, wy: float):
        nx = 2.0 * (wx - d_min[0]) / d_range[0] - 1.0
        ny = 2.0 * (wy - d_min[1]) / d_range[1] - 1.0
        return nx, ny

    # Half-extents of one cell in normalised space
    half_w = cell_size / d_range[0]
    half_h = cell_size / d_range[1]

    walls = []
    for r in range(n_rows):
        for c in range(n_cols):
            if maze_map[r][c] == 1:
                # World-coordinate centre of cell (r, c)
                wx = (c - (n_cols - 1) / 2.0) * cell_size
                wy = ((n_rows - 1) / 2.0 - r)  * cell_size
                cx, cy = to_norm(wx, wy)
                walls.append({'cx': cx, 'cy': cy, 'hw': half_w, 'hh': half_h})

    return {
        'walls':    walls,
        'view_min': -1.9,
        'view_max':  1.9,
        'maze_map': maze_map,
        'd_min':    d_min,
        'd_max':    d_max,
    }


# ---------------------------------------------------------------------------
# /api/run   — run both Plain DPM and Safe DPM from identical prior
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    model_name:  str   = 've_unet_umaze_diffuser.pt'
    n_steps:     int   = 20
    c:           float = 1.0
    k1:          float = 1.0
    k2:          float = 1.0
    r:           float = 0.1
    gamma_delta: float = 0.0
    obs_x:       float = 0.0
    obs_y:       float = 0.0
    start_x:     float = -0.8
    start_y:     float = -0.8
    goal_x:      float = 0.8
    goal_y:      float = 0.8
    alpha0:      float = 1.0
    use_softplus: bool = True
    seed:        Optional[int] = None


@app.post('/api/run')
def run_optimisation(req: RunRequest):
    """
    Run both Plain DPM-Solver-1 and Safe DPM-Solver-1 from the **exact same**
    initial noise prior.  Returns per-step trajectory history for both runs,
    plus per-step CBF metrics for the safe trajectory.

    Response shape:
        n_steps          : int
        T_steps          : int
        start            : [2]
        goal             : [2]
        prior            : [T, 2]             — shared step-0 noise
        plain_history    : [n_steps+1, T, 2]
        safe_history     : [n_steps+1, T, 2]
        cbf_step_data    : [n_steps+1] dicts  — h_Xt, d_raw, d_tilde, h_wi, sigma_i, grad_x, grad_y
    """
    try:
        ema_model, ve, T_steps = _load_model(req.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Model load error: {exc}')

    # Clamp k2 to avoid division by zero
    k2 = max(req.k2, 1e-3)

    # Build obstacle tensor [1, 3]
    obstacles = torch.tensor(
        [[req.obs_x, req.obs_y, req.r]], dtype=torch.float32, device=DEVICE
    )

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
        # --- Plain DPM-Solver-1 ---
        t0 = time.perf_counter()
        _, plain_history = dpm_solver_1_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            T_steps=T_steps, n_steps=req.n_steps,
            device=DEVICE, x_init=x_init.clone(),
            return_history=True,
        )
        plain_time = time.perf_counter() - t0

        # --- Safe DPM-Solver-1 + CBF ---
        t0 = time.perf_counter()
        _, safe_history, _before_history, cbf_history = dpm_solver_1_cbf_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            obstacles=obstacles,
            T_steps=T_steps, n_steps=req.n_steps,
            k1=req.k1, k2=k2, c=req.c,
            alpha0=req.alpha0, gamma_delta=req.gamma_delta, use_softplus=req.use_softplus,
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
        'before_history': [traj_to_list(h) for h in _before_history],
        'safe_history':   [traj_to_list(h) for h in safe_history],
        'cbf_step_data':  cbf_history,
        'plain_time':     round(plain_time, 3),
        'safe_time':      round(safe_time,  3),
    }


# ---------------------------------------------------------------------------
# /api/math   — live re-evaluation for a given traj + parameters
# ---------------------------------------------------------------------------

class MathRequest(BaseModel):
    traj:        List[List[float]]   # [T, 2]
    c:           float = 1.0
    k1:          float = 1.0
    k2:          float = 1.0
    r:           float = 0.1
    gamma_delta: float = 0.0
    obs_x:       float = 0.0
    obs_y:       float = 0.0
    use_softplus: bool = True


@app.post('/api/math')
def evaluate_math(req: MathRequest):
    """
    Re-evaluate CBF metrics for a user-provided trajectory and parameters.
    Called on parameter-slider changes or waypoint drags (no re-sampling).

    Response: same dict as cbf_step_data entries in /api/run.
    """
    k2 = max(req.k2, 1e-3)

    Xt = torch.tensor(req.traj, dtype=torch.float32, device=DEVICE)  # [T, 2]
    obstacles = torch.tensor(
        [[req.obs_x, req.obs_y, req.r]], dtype=torch.float32, device=DEVICE
    )

    try:
        metrics = compute_cbf_metrics(Xt, obstacles, req.c, req.k1, k2, req.gamma_delta, req.use_softplus)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'CBF math error: {exc}')

    return metrics


# ---------------------------------------------------------------------------
# /api/recompute_ctrl  — recompute CBF metrics + control + after trajectory
# ---------------------------------------------------------------------------

class RecomputeRequest(BaseModel):
    before_traj:  List[List[float]]
    eps_pred_x:   List[float]
    eps_pred_y:   List[float]
    sigma_delta:  float
    sigma_dot:    float
    noise_idx:    int
    step_delta:   int   = 0
    n_steps:      int
    c:            float = 1.0
    k1:           float = 1.0
    k2:           float = 1.0
    r:            float = 0.1
    gamma_delta:  float = 0.0
    obs_x:        float = 0.0
    obs_y:        float = 0.0
    alpha0:       float = 1.0
    use_softplus: bool  = True


@app.post('/api/recompute_ctrl')
def recompute_ctrl_endpoint(req: RecomputeRequest):
    """
    Recompute CBF metrics and after-control trajectory for one step,
    given a (possibly modified) before-control trajectory and cached eps_pred.
    Does not re-run the score network.

    Response: cbf_history entry dict PLUS 'after_traj' [[x, y], ...]
    """
    k2 = max(req.k2, 1e-3)

    obstacles = torch.tensor(
        [[req.obs_x, req.obs_y, req.r]], dtype=torch.float32, device=DEVICE
    )

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
