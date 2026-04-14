"""
samplers_ellipsoids_cfg.py — DPM-Solver-1 samplers with CFG + ellipsoidal CBF.

Extends EllipsoidalCBFSampling/models/samplers_ellipses.py to support
Classifier-Free Guidance (CFG) for ellipsoid obstacle conditioning.

Provides:
    dpm_solver_1_cfg_sample        — plain CFG ODE (no CBF safety)
    dpm_solver_1_cbf_cfg_sample    — CFG ODE + trajectory-level ellipsoidal CBF
    recompute_cbf_step             — recompute CBF metrics for modified trajectory
                                     (unchanged; used by the interactive visualiser)

CFG inference:
    Per denoising step, two forward passes are made:
        eps_cond   = model(x, sigma, x_start, x_goal, ellipsoids)
        eps_uncond = model(x, sigma, x_start, x_goal, zeros_like(ellipsoids))
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    When guidance_scale=1.0, this reduces to the conditional prediction only.
    When guidance_scale=0.0, this reduces to the unconditional prediction.

CBF correction (dpm_solver_1_cbf_cfg_sample):
    The CBF uses the raw ellipsoid geometry (obstacles [N, 4]) independently of
    the learned conditioning. The guided eps_pred is passed to _cbf_control_term,
    which computes the safety correction as in samplers_ellipses.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from EllipsoidalCBFSampling.CBF.trajectory_cbf_ellipses import (
    trajectory_cbf, grad_hXt_dXt, compute_cbf_metrics,
)


# ---------------------------------------------------------------------------
# Plain CFG DPM-Solver-1  (no CBF)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpm_solver_1_cfg_sample(
    model,
    ve_diffusion,
    x_start:        torch.Tensor,       # [2] or [B, 2]
    x_goal:         torch.Tensor,       # [2] or [B, 2]
    ellipsoids:     torch.Tensor,       # [B, 5, 4]  (cx,cy,a,b); zeros = null
    T_steps:        int   = 64,
    n_steps:        int   = 25,
    guidance_scale: float = 3.0,
    device:         str   = 'cpu',
    x_init:         torch.Tensor = None,
    return_history: bool  = False,
):
    """
    DPM-Solver-1 with Classifier-Free Guidance for ellipsoid conditioning.

        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        x_{i+1}   = x_i - (sigma_i - sigma_{i+1}) * eps_guided

    Args:
        ellipsoids    : [B, 5, 4] obstacle set; pass zeros([B,5,4]) for unconditional.
        guidance_scale: CFG weight w (>=1 strengthens conditioning; 1 = conditional only).
        x_init        : optional shared prior [B, T, 2].
        return_history: if True, also return list of [B, T, 2] at each step.

    Returns:
        x        : [B, T_steps, 2]
        history  : list of [B, T, 2] (length n_steps+1), only if return_history.
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start    = x_start.to(device)
    x_goal     = x_goal.to(device)
    ellipsoids = ellipsoids.to(device)

    sigmas    = ve_diffusion.sigmas.to(device)
    N_lvl     = ve_diffusion.n_levels
    indices   = torch.linspace(N_lvl, 0, n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]

    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    null_ellipsoids = torch.zeros_like(ellipsoids)
    history = [x.clone()] if return_history else None

    for i in range(n_steps):
        sig_cur  = sigma_seq[i]
        sig_next = sigma_seq[i + 1]
        sig_b    = sig_cur.expand(B)

        # Two model calls for CFG
        eps_cond   = model(x, sig_b, x_start, x_goal, ellipsoids)
        eps_uncond = model(x, sig_b, x_start, x_goal, null_ellipsoids)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x = x - (sig_cur - sig_next) * eps_guided

        x[:, 0, :]  = x_start
        x[:, -1, :] = x_goal

        if return_history:
            history.append(x.clone())

    if return_history:
        return x, history
    return x


# ---------------------------------------------------------------------------
# CBF control term  (VE case, eq. 75-76 — identical to samplers_ellipses.py)
# ---------------------------------------------------------------------------

def _cbf_control_term(
    Xt:            torch.Tensor,   # [T, 2]  single trajectory
    eps_pred:      torch.Tensor,   # [T, 2]  guided score network output
    sig_cur:       float,
    sig_next:      float,
    sigma_dot:     float,
    noise_idx:     int,
    noise_idx_next: int,
    n_steps:       int,
    obstacles:     torch.Tensor,   # [N, 4]  (cx,cy,a,b)
    k1:            float,
    k2:            float,
    c:             float,
    alpha0:        float,
    gamma_delta:   float,
    use_softplus:  bool = True,
    eps:           float = 1e-8,
) -> tuple:
    """
    Compute CBF control correction for a single trajectory [T, 2].

    Uses the guided eps_pred (from CFG) as the effective score prediction.
    Correction formula (VE case):
        omega       = sigma_dot * (grad_h · eps_pred) + alpha0 * h
        ctrl        = min(0, omega) / (||grad_h||^2 + 1/rho * h^2 + eps) * grad_h

    Returns:
        ctrl        : [T, 2]
        omega_val   : float
        ctrl_raw    : [T, 2]
        sigma_delta : float
    """
    h_val  = trajectory_cbf(Xt, obstacles, k1, k2, gamma_delta, c, use_softplus)
    grad_h = grad_hXt_dXt(Xt, obstacles, k1, k2, gamma_delta, c, use_softplus)  # [T, 2]

    sigma_delta = float(sig_cur - sig_next)
    step_delta  = noise_idx - noise_idx_next

    omega     = (sigma_delta * grad_h * eps_pred).sum() + step_delta * alpha0 * h_val
    omega_neg = torch.clamp(omega, max=0.0)

    rho = max(0.0, noise_idx - n_steps / 2.0)
    if rho > 0.0:
        denom = (grad_h * grad_h).sum() + (1.0 / rho) * h_val ** 2 + eps
    else:
        denom = (grad_h * grad_h).sum() + eps

    ctrl_raw = -(omega_neg / denom) * grad_h   # [T, 2]
    ctrl     = ctrl_raw
    return ctrl, omega_neg.item(), ctrl_raw, sigma_delta


# ---------------------------------------------------------------------------
# Recompute CBF step  (for interactive visualiser — unchanged API)
# ---------------------------------------------------------------------------

def recompute_cbf_step(
    before_traj:  list,
    eps_pred_x:   list,
    eps_pred_y:   list,
    sigma_delta:  float,
    sigma_dot:    float,
    noise_idx:    int,
    step_delta:   int,
    n_steps:      int,
    obstacles:    torch.Tensor,   # [N, 4]
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

    API identical to samplers_ellipses.recompute_cbf_step.
    """
    T  = len(before_traj)
    Xt = torch.tensor(before_traj, dtype=torch.float32, device=device)
    ep = torch.zeros(T, 2, dtype=torch.float32, device=device)
    ep[:, 0] = torch.tensor(eps_pred_x, dtype=torch.float32, device=device)
    ep[:, 1] = torch.tensor(eps_pred_y, dtype=torch.float32, device=device)

    if obstacles.shape[0] == 0 or sigma_dot == 0.0:
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
        sig_cur=sigma_delta, sig_next=0.0,
        sigma_dot=sigma_dot,
        noise_idx=noise_idx,
        noise_idx_next=noise_idx + step_delta,
        n_steps=n_steps,
        obstacles=obstacles,
        k1=k1, k2=k2, c=c, alpha0=alpha0,
        gamma_delta=gamma_delta, use_softplus=use_softplus,
    )
    after_Xt = Xt + ctrl

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
# CFG + CBF safe DPM-Solver-1
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpm_solver_1_cbf_cfg_sample(
    model,
    ve_diffusion,
    x_start:        torch.Tensor,       # [2] or [B, 2]
    x_goal:         torch.Tensor,       # [2] or [B, 2]
    ellipsoids:     torch.Tensor,       # [B, 5, 4]  learned conditioning
    obstacles:      torch.Tensor,       # [N, 4]     raw geometry for CBF (cx,cy,a,b)
    T_steps:        int   = 64,
    n_steps:        int   = 25,
    guidance_scale: float = 3.0,
    k1:             float = 1.0,
    k2:             float = 1.0,
    c:              float = 1.0,
    alpha0:         float = 1.0,
    gamma_delta:    float = 0.0,
    use_softplus:   bool  = True,
    device:         str   = 'cpu',
    x_init:         torch.Tensor = None,
    return_history: bool  = False,
):
    """
    DPM-Solver-1 with Classifier-Free Guidance + trajectory-level CBF correction.

    Per denoising step:
        1. Two model calls (conditional + unconditional) → CFG-blended eps_guided
        2. Base ODE step: x_new = x - (sig_cur - sig_next) * eps_guided
        3. CBF correction applied to x_new using raw obstacle geometry
        4. Inpainting: re-pin start and goal waypoints

    Args:
        ellipsoids    : [B, 5, 4]  — learned conditioning (for model forward pass)
        obstacles     : [N, 4]     — raw ellipsoid geometry for CBF (cx,cy,a,b)
                        Pass torch.zeros(0, 4) to disable CBF.
        guidance_scale: CFG blending weight (>=1 recommended).
        return_history: if True, return (x, traj_history, before_history, cbf_history).

    Returns (return_history=False):
        x : [B, T_steps, 2]

    Returns (return_history=True):
        x              : [B, T_steps, 2]
        traj_history   : list[n_steps+1] of [B, T, 2]  — after CBF ctrl
        before_history : list[n_steps+1] of [B, T, 2]  — before CBF ctrl
        cbf_history    : list[n_steps+1] of dict        — CBF metrics (batch 0)
    """
    if x_start.dim() == 1:
        x_start = x_start.unsqueeze(0)
        x_goal  = x_goal.unsqueeze(0)
    B = x_start.shape[0]
    x_start    = x_start.to(device)
    x_goal     = x_goal.to(device)
    ellipsoids = ellipsoids.to(device)
    obstacles  = obstacles.to(device)

    sigmas    = ve_diffusion.sigmas.to(device)
    N_lvl     = ve_diffusion.n_levels
    indices   = torch.linspace(N_lvl, 0, n_steps + 1).long().clamp(0, N_lvl)
    sigma_seq = sigmas[indices]

    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.randn(B, T_steps, 2, device=device) * sigma_seq[0]

    null_ellipsoids = torch.zeros_like(ellipsoids)
    has_obstacles   = obstacles.shape[0] > 0

    traj_history   = [x.clone()] if return_history else None
    before_history = [x.clone()] if return_history else None
    cbf_history    = [] if return_history else None

    if return_history:
        m0 = (compute_cbf_metrics(x[0], obstacles, c, k1, k2, gamma_delta)
              if has_obstacles else _empty_cbf_metrics(T_steps))
        m0.update({
            'omega': None, 'ctrl_x': [0.0]*T_steps, 'ctrl_y': [0.0]*T_steps,
            'ctrl_raw_x': [0.0]*T_steps, 'ctrl_raw_y': [0.0]*T_steps,
            'sigma_delta': 0.0, 'eps_pred_x': [0.0]*T_steps,
            'eps_pred_y': [0.0]*T_steps, 'sigma_dot': 0.0,
            'noise_idx': 0, 'step_delta': 0,
        })
        cbf_history.append(m0)

    for step_i in range(n_steps):
        sig_cur  = sigma_seq[step_i]
        sig_next = sigma_seq[step_i + 1]
        sig_b    = sig_cur.expand(B)

        # CFG: two model calls, blend
        eps_cond   = model(x, sig_b, x_start, x_goal, ellipsoids)
        eps_uncond = model(x, sig_b, x_start, x_goal, null_ellipsoids)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # [B, T, 2]

        # Base ODE step ("before control" trajectory)
        x_new = x - (sig_cur - sig_next) * eps_guided

        if return_history:
            before_history.append(x_new.clone())

        # CBF correction per sample
        ctrl_b0, omega_b0, ctrl_raw_b0, sigma_delta_b0 = None, None, None, None
        eps_pred_b0, sigma_dot_val, noise_idx_val, step_delta_val = None, None, None, None

        if has_obstacles:
            sigma_dot_val  = ve_diffusion.sigma_dot(sig_cur).item()
            noise_idx_val  = indices[step_i].item()
            noise_idx_next = indices[step_i + 1].item()
            step_delta_val = noise_idx_next - noise_idx_val

            for b in range(B):
                ctrl, omega_val, ctrl_raw, sigma_delta = _cbf_control_term(
                    x[b], eps_guided[b],
                    sig_cur.item(), sig_next.item(), sigma_dot_val,
                    noise_idx_val, noise_idx_next, n_steps,
                    obstacles, k1, k2, c, alpha0, gamma_delta, use_softplus,
                )
                x_new[b] = x_new[b] + ctrl
                if b == 0:
                    ctrl_b0        = ctrl.clone()
                    omega_b0       = omega_val
                    ctrl_raw_b0    = ctrl_raw.clone()
                    sigma_delta_b0 = sigma_delta
                    eps_pred_b0    = eps_guided[0].clone()

        x_new[:, 0, :]  = x_start
        x_new[:, -1, :] = x_goal
        x = x_new

        if return_history:
            traj_history.append(x.clone())

            if has_obstacles:
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
