"""
Modular DPM samplers for VE-SDE trajectory generation.

Extracted from notebooks/train_and_sample_circles_copy.ipynb.

Provides:
    dpm_solver_1_sample        — plain deterministic ODE (no safety)
    dpm_solver_1_cbf_sample    — safe ODE with trajectory-level CBF correction
    recompute_cbf_step         — recompute CBF metrics + ctrl for a modified trajectory

Both accept return_history=True to capture per-step data for the visualiser.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Ensure project root is on sys.path for sibling-package imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from CBF.trajectory_cbf import trajectory_cbf, grad_hXt_dXt, compute_cbf_metrics


# ---------------------------------------------------------------------------
# Plain DPM-Solver-1  (deterministic ODE, no CBF)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpm_solver_1_sample(
    model,
    ve_diffusion,
    x_start: torch.Tensor,       # [2] or [B, 2]
    x_goal: torch.Tensor,        # [2] or [B, 2]
    T_steps: int = 64,
    n_steps: int = 25,
    device: str = 'cpu',
    x_init: torch.Tensor = None, # [B, T, 2] shared prior; sampled fresh if None
    return_history: bool = False,
):
    """
    DPM-Solver-1: first-order deterministic ODE for VE-SDE.

        x_{i+1} = x_i - (sigma_i - sigma_{i+1}) * eps_theta(x_i, sigma_i)

    Args:
        model         : score network (ema recommended)
        ve_diffusion  : VEDiffusion instance
        x_start       : [2] or [B, 2]
        x_goal        : [2] or [B, 2]
        T_steps       : number of waypoints
        n_steps       : denoising steps
        device        : torch device string
        x_init        : optional shared prior [B, T, 2]
        return_history: if True, also return list of [B, T, 2] at each step

    Returns:
        x             : [B, T_steps, 2]  final trajectory
        history       : list of [B, T, 2] (length n_steps+1), only if return_history
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start = x_start.to(device)
    x_goal  = x_goal.to(device)

    sigmas    = ve_diffusion.sigmas.to(device)
    N_lvl     = ve_diffusion.n_levels
    indices   = torch.linspace(N_lvl, 0, n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]   # [n_steps+1], decreasing

    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    history = [x.clone()] if return_history else None

    for i in range(n_steps):
        sig_cur  = sigma_seq[i]
        sig_next = sigma_seq[i + 1]

        eps_pred = model(x, sig_cur.expand(B), x_start, x_goal)  # [B, T, 2]
        x = x - (sig_cur - sig_next) * eps_pred

        # Inpainting: re-pin start and goal waypoints
        x[:, 0, :]  = x_start
        x[:, -1, :] = x_goal

        if return_history:
            history.append(x.clone())

    if return_history:
        return x, history
    return x


# ---------------------------------------------------------------------------
# CBF control term  (VE case, eq. 75-76)
# ---------------------------------------------------------------------------

def _cbf_control_term(
    Xt: torch.Tensor,         # [T, 2]  single trajectory
    eps_pred: torch.Tensor,   # [T, 2]  score network output
    sig_cur: float,
    sig_next: float,
    sigma_dot: float,
    noise_idx: int,           # current noise level index (for regularisation)
    noise_idx_next: int,
    n_steps: int,
    obstacles: torch.Tensor,  # [N, 3]
    k1: float,
    k2: float,
    c: float,
    alpha0: float,
    gamma_delta: float,
    use_softplus: bool = True,
    eps: float = 1e-8,
) -> tuple:
    """
    Compute CBF control correction for a single trajectory [T, 2].

    VE-case safe DPM-Solver-1 update (eq. 75):
        ctrl = min(0, omega) / (||grad_h||^2 + 1/rho * h^2 + eps) * grad_h * sigma_delta

    where:
        omega       = sigma_dot * (grad_h · eps_pred) + alpha0 * h
        sigma_delta = sig_cur - sig_next
        rho         = max(0, noise_idx - n_steps/2)   [regularisation schedule]

    Returns:
        ctrl        : [T, 2]  — subtract this from x_new
        omega_val   : float   — scalar omega (for inspector display)
        ctrl_raw    : [T, 2]  — ctrl before multiplying by sigma_delta
    """
    h_val  = trajectory_cbf(Xt, obstacles, k1, k2, gamma_delta, c, use_softplus)
    grad_h = grad_hXt_dXt(Xt, obstacles, k1, k2, gamma_delta, c, use_softplus)  # [T, 2]


    sigma_delta = float(sig_cur - sig_next)
    step_delta =   noise_idx - noise_idx_next  # should be positive, but just in case of weird scheduler


    omega     = (sigma_delta * grad_h * eps_pred).sum() + step_delta*alpha0 * h_val
    omega_neg = torch.clamp(omega, max=0.0)

    rho = max(0.0, noise_idx - n_steps / 4.0)
    if rho > 0.0:
        denom = (grad_h * grad_h).sum() + (1.0 / rho) * h_val ** 2 + eps
    else:
        denom = (grad_h * grad_h).sum() + eps

    ctrl_raw    = (omega_neg / denom) * grad_h          # [T, 2]  before sigma_delta
    ctrl        = ctrl_raw                # [T, 2]  final correction
    return ctrl, omega.item(), ctrl_raw, sigma_delta


# ---------------------------------------------------------------------------
# Recompute CBF step  (for interactive visualiser — no score network re-run)
# ---------------------------------------------------------------------------

def recompute_cbf_step(
    before_traj:  list,        # [T, 2] as Python list-of-lists (possibly user-modified)
    eps_pred_x:   list,        # [T] floats — cached from original run
    eps_pred_y:   list,        # [T] floats
    sigma_delta:  float,
    sigma_dot:    float,
    noise_idx:    int,
    step_delta:   int,
    n_steps:      int,
    obstacles:    torch.Tensor,  # [N, 3]
    k1:           float,
    k2:           float,
    c:            float,
    alpha0:       float,
    gamma_delta:  float,
    use_softplus: bool = True,
    device:       str  = 'cpu',
) -> dict:
    """
    Recompute CBF metrics and after-control trajectory for a single denoising
    step, given a (possibly user-modified) before-control trajectory and the
    cached eps_pred from the original score-network call.

    Uses sig_cur=sigma_delta, sig_next=0.0 as a dummy: only their difference
    (= sigma_delta) is used inside _cbf_control_term.

    Returns a dict with all cbf_history fields PLUS:
        'after_traj': [[x, y], ...] list of T waypoints  (before_traj - ctrl)
    """
    T  = len(before_traj)
    Xt = torch.tensor(before_traj, dtype=torch.float32, device=device)   # [T, 2]
    ep = torch.zeros(T, 2, dtype=torch.float32, device=device)
    ep[:, 0] = torch.tensor(eps_pred_x, dtype=torch.float32, device=device)
    ep[:, 1] = torch.tensor(eps_pred_y, dtype=torch.float32, device=device)

    if obstacles.shape[0] == 0 or sigma_dot == 0.0:
        # Step 0 or no obstacles: no control
        m = compute_cbf_metrics(Xt, obstacles, c, k1, k2, gamma_delta, use_softplus)
        m['omega']       = None
        m['ctrl_x']      = [0.0] * T
        m['ctrl_y']      = [0.0] * T
        m['ctrl_raw_x']  = [0.0] * T
        m['ctrl_raw_y']  = [0.0] * T
        m['sigma_delta'] = sigma_delta
        m['eps_pred_x']  = list(eps_pred_x)
        m['eps_pred_y']  = list(eps_pred_y)
        m['sigma_dot']   = sigma_dot
        m['noise_idx']   = noise_idx
        m['step_delta']  = step_delta
        m['after_traj']  = before_traj
        return m

    ctrl, omega_val, ctrl_raw, _ = _cbf_control_term(
        Xt, ep,
        sig_cur=sigma_delta, sig_next=0.0,    # dummy: only difference is used
        sigma_dot=sigma_dot,
        noise_idx=noise_idx,
        noise_idx_next=noise_idx + step_delta,
        n_steps=n_steps,
        obstacles=obstacles,
        k1=k1, k2=k2, c=c, alpha0=alpha0,
        gamma_delta=gamma_delta, use_softplus=use_softplus,
    )
    after_Xt = Xt - ctrl  # [T, 2]

    m = compute_cbf_metrics(Xt, obstacles, c, k1, k2, gamma_delta, use_softplus)
    m['omega']       = omega_val
    m['ctrl_x']      = ctrl[:, 0].tolist()
    m['ctrl_y']      = ctrl[:, 1].tolist()
    m['ctrl_raw_x']  = ctrl_raw[:, 0].tolist()
    m['ctrl_raw_y']  = ctrl_raw[:, 1].tolist()
    m['sigma_delta'] = sigma_delta
    m['eps_pred_x']  = list(eps_pred_x)
    m['eps_pred_y']  = list(eps_pred_y)
    m['sigma_dot']   = sigma_dot
    m['noise_idx']   = noise_idx
    m['step_delta']  = step_delta
    m['after_traj']  = after_Xt.tolist()
    return m


# ---------------------------------------------------------------------------
# Safe DPM-Solver-1  (deterministic ODE + CBF correction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpm_solver_1_cbf_sample(
    model,
    ve_diffusion,
    x_start: torch.Tensor,        # [2] or [B, 2]
    x_goal: torch.Tensor,         # [2] or [B, 2]
    obstacles: torch.Tensor,      # [N, 3]  (px, py, r)
    T_steps: int = 64,
    n_steps: int = 25,
    k1: float = 1.0,
    k2: float = 1.0,
    c: float = 1.0,
    alpha0: float = 1.0,
    gamma_delta: float = 0.0,
    use_softplus: bool = True,
    device: str = 'cpu',
    x_init: torch.Tensor = None,  # [B, T, 2] shared prior
    return_history: bool = False,
):
    """
    DPM-Solver-1 with trajectory-level CBF safety correction.

    At each denoising step i:
        x_new = x - (sig_cur - sig_next) * eps_theta      # base ODE  ("before control")
              - ctrl                                       # CBF correction (eq. 75)

    The CBF correction is zero when omega >= 0 (constraint already satisfied).

    Args:
        obstacles     : [N, 3]. Pass torch.zeros(0, 3) to disable CBF (= plain sampler).
        x_init        : optional shared prior [B, T, 2] — use the same noise as plain run.
        return_history: if True, return (x, traj_history, before_history, cbf_metrics_history)

    Returns (return_history=False):
        x : [B, T_steps, 2]

    Returns (return_history=True):
        x              : [B, T_steps, 2]
        traj_history   : list[n_steps+1] of [B, T, 2]  — trajectory after CBF ctrl
        before_history : list[n_steps+1] of [B, T, 2]  — trajectory before CBF ctrl (ODE result)
        cbf_history    : list[n_steps+1] of dict        — CBF metrics per step (batch 0,
                         computed from before_history[i][0])
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start   = x_start.to(device)
    x_goal    = x_goal.to(device)
    obstacles = obstacles.to(device)

    sigmas    = ve_diffusion.sigmas.to(device)
    N_lvl     = ve_diffusion.n_levels
    indices   = torch.linspace(N_lvl, 0, n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]

    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    has_obstacles = obstacles.shape[0] > 0

    traj_history   = [x.clone()] if return_history else None
    before_history = [x.clone()] if return_history else None   # step 0: prior = before = after
    cbf_history    = [] if return_history else None

    if return_history:
        # Step 0: pure noise prior — no ODE step yet, no control applied
        m0 = (compute_cbf_metrics(x[0], obstacles, c, k1, k2, gamma_delta)
              if has_obstacles else _empty_cbf_metrics(T_steps))
        m0['omega']       = None
        m0['ctrl_x']      = [0.0] * T_steps
        m0['ctrl_y']      = [0.0] * T_steps
        m0['ctrl_raw_x']  = [0.0] * T_steps
        m0['ctrl_raw_y']  = [0.0] * T_steps
        m0['sigma_delta'] = 0.0
        m0['eps_pred_x']  = [0.0] * T_steps
        m0['eps_pred_y']  = [0.0] * T_steps
        m0['sigma_dot']   = 0.0
        m0['noise_idx']   = 0
        m0['step_delta']  = 0
        cbf_history.append(m0)

    for step_i in range(n_steps):
        sig_cur  = sigma_seq[step_i]
        sig_next = sigma_seq[step_i + 1]

        eps_pred = model(x, sig_cur.expand(B), x_start, x_goal)  # [B, T, 2]

        # Base DPM-Solver-1 step  ("before control" trajectory)
        x_new = x - (sig_cur - sig_next) * eps_pred

        # Snapshot BEFORE CBF correction is applied
        if return_history:
            before_history.append(x_new.clone())   # [B, T, 2]

        # CBF correction per sample — capture b=0 values for inspector
        ctrl_b0, omega_b0, ctrl_raw_b0, sigma_delta_b0 = None, None, None, None
        eps_pred_b0, sigma_dot_val, noise_idx_val, step_delta_val = None, None, None, None

        if has_obstacles:
            sigma_dot_val  = ve_diffusion.sigma_dot(sig_cur).item()
            noise_idx_val  = indices[step_i].item()
            noise_idx_next = indices[step_i + 1].item()
            step_delta_val = noise_idx_next - noise_idx_val

            for b in range(B):
                ctrl, omega_val, ctrl_raw, sigma_delta = _cbf_control_term(
                    x[b], eps_pred[b],
                    sig_cur.item(), sig_next.item(), sigma_dot_val,
                    noise_idx_val, noise_idx_next, n_steps,
                    obstacles, k1, k2, c, alpha0, gamma_delta, use_softplus,
                )
                x_new[b] = x_new[b] - ctrl
                if b == 0:
                    ctrl_b0        = ctrl.clone()
                    omega_b0       = omega_val
                    ctrl_raw_b0    = ctrl_raw.clone()
                    sigma_delta_b0 = sigma_delta
                    eps_pred_b0    = eps_pred[0].clone()

        # Inpainting: re-pin start and goal waypoints
        x_new[:, 0, :]  = x_start
        x_new[:, -1, :] = x_goal

        x = x_new

        if return_history:
            traj_history.append(x.clone())

            if has_obstacles:
                # Compute metrics from before-control trajectory (before_history[-1][0])
                m = compute_cbf_metrics(
                    before_history[-1][0], obstacles, c, k1, k2, gamma_delta, use_softplus
                )
                m['omega']       = omega_b0
                m['ctrl_x']      = ctrl_b0[:, 0].tolist()
                m['ctrl_y']      = ctrl_b0[:, 1].tolist()
                m['ctrl_raw_x']  = ctrl_raw_b0[:, 0].tolist()
                m['ctrl_raw_y']  = ctrl_raw_b0[:, 1].tolist()
                m['sigma_delta'] = sigma_delta_b0
                m['eps_pred_x']  = eps_pred_b0[:, 0].tolist()
                m['eps_pred_y']  = eps_pred_b0[:, 1].tolist()
                m['sigma_dot']   = sigma_dot_val
                m['noise_idx']   = noise_idx_val
                m['step_delta']  = step_delta_val
            else:
                m = _empty_cbf_metrics(T_steps)
            cbf_history.append(m)

    if return_history:
        return x, traj_history, before_history, cbf_history
    return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_cbf_metrics(T: int) -> dict:
    return {
        'h_Xt':       0.0,
        'd_raw':      [0.0] * T,
        'd_tilde':    [0.0] * T,
        'h_wi':       [0.0] * T,
        'sigma_i':    [1.0 / T] * T,
        'grad_x':     [0.0] * T,
        'grad_y':     [0.0] * T,
        'omega':      None,
        'ctrl_x':     [0.0] * T,
        'ctrl_y':     [0.0] * T,
        'ctrl_raw_x': [0.0] * T,
        'ctrl_raw_y': [0.0] * T,
        'sigma_delta': 0.0,
        'eps_pred_x':  [0.0] * T,
        'eps_pred_y':  [0.0] * T,
        'sigma_dot':   0.0,
        'noise_idx':   0,
        'step_delta':  0,
    }
