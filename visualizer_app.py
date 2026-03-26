"""
SafeDPMSolver Interactive Visualiser — FastAPI Backend
======================================================
Run with:
    conda run -n py_3_10 uvicorn visualizer_app:app --port 8001 --reload

Endpoints:
    GET  /                  -> redirect to /static/index.html
    GET  /api/models        -> list available .pt checkpoints
    POST /api/run           -> run Plain + Safe DPM from identical prior, return full step history
    POST /api/math          -> re-evaluate CBF metrics for a given trajectory + params
"""

from __future__ import annotations

import copy
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
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.score_net import TemporalUnet
from models.ve_diffusion import VEDiffusion
from models.samplers import dpm_solver_1_sample, dpm_solver_1_cbf_sample
from CBF.trajectory_cbf import compute_cbf_metrics

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
app = FastAPI(title='SafeDPMSolver Visualiser', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(PROJECT_DIR / 'static')), name='static')


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/static/index.html')


# ---------------------------------------------------------------------------
# Model cache  (loaded lazily, kept in memory)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _load_model(model_name: str):
    """Load and cache (ema_model, ve_diffusion) for a given checkpoint name."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_path = CHECKPOINTS_DIR / model_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)

    score_net = TemporalUnet(**ckpt['model_cfg']).to(DEVICE)
    ema_model = copy.deepcopy(score_net).to(DEVICE)
    ema_model.load_state_dict(ckpt['ema_model'])
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    ve = VEDiffusion(model=ema_model, **ckpt['ve_cfg']).to(DEVICE)

    _model_cache[model_name] = (ema_model, ve)
    return ema_model, ve


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
# /api/run   — run both Plain DPM and Safe DPM from identical prior
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    model_name:  str   = 've_unet_circles_100k.pt'
    n_steps:     int   = 20
    c:           float = 1.0
    k1:          float = 1.0
    k2:          float = 1.0
    r:           float = 0.3
    gamma_delta: float = 0.05
    obs_x:       float = 0.0
    obs_y:       float = 0.0
    start_x:     float = -0.8
    start_y:     float = -0.8
    goal_x:      float = 0.8
    goal_y:      float = 0.8
    alpha0:      float = 1.0
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
        ema_model, ve = _load_model(req.model_name)
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

    # Determine T_steps from model config
    ckpt_path = CHECKPOINTS_DIR / req.model_name
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    T_steps = ckpt['model_cfg'].get('T_steps', 64)

    # Sample shared prior
    if req.seed is not None:
        torch.manual_seed(req.seed)
    sigmas    = ve.sigmas.to(DEVICE)
    N_lvl     = ve.n_levels
    indices   = torch.linspace(N_lvl, 0, req.n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]
    x_init    = torch.randn(1, T_steps, 2, device=DEVICE) * sigma_seq[0]

    try:
        # --- Plain DPM-Solver-1 ---
        _, plain_history = dpm_solver_1_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            T_steps=T_steps, n_steps=req.n_steps,
            device=DEVICE, x_init=x_init.clone(),
            return_history=True,
        )

        # --- Safe DPM-Solver-1 + CBF ---
        _, safe_history, cbf_history = dpm_solver_1_cbf_sample(
            ema_model, ve,
            x_start=x_start, x_goal=x_goal,
            obstacles=obstacles,
            T_steps=T_steps, n_steps=req.n_steps,
            k1=req.k1, k2=k2, c=req.c,
            alpha0=req.alpha0, gamma_delta=req.gamma_delta,
            device=DEVICE, x_init=x_init.clone(),
            return_history=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Sampler error: {exc}')

    def traj_to_list(t):
        # t: [1, T, 2] -> [[x,y], ...]
        return t[0].cpu().tolist()

    return {
        'n_steps':       req.n_steps,
        'T_steps':       T_steps,
        'start':         [req.start_x, req.start_y],
        'goal':          [req.goal_x,  req.goal_y],
        'prior':         traj_to_list(x_init),
        'plain_history': [traj_to_list(h) for h in plain_history],
        'safe_history':  [traj_to_list(h) for h in safe_history],
        'cbf_step_data': cbf_history,
    }


# ---------------------------------------------------------------------------
# /api/math   — live re-evaluation for a given traj + parameters
# ---------------------------------------------------------------------------

class MathRequest(BaseModel):
    traj:        List[List[float]]   # [T, 2]
    c:           float = 1.0
    k1:          float = 1.0
    k2:          float = 1.0
    r:           float = 0.3
    gamma_delta: float = 0.05
    obs_x:       float = 0.0
    obs_y:       float = 0.0


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
        metrics = compute_cbf_metrics(Xt, obstacles, req.c, req.k1, k2, req.gamma_delta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'CBF math error: {exc}')

    return metrics
