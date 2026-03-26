"""
Trajectory-level Control Barrier Functions for circular obstacles.

Reference: SafeDiffusionDerivationsWithSoftplus.pdf, Section 4

Updated formulation uses softplus distance transformation for:
  - Numerical stability: subtracting minima before logsumexp prevents overflow/underflow
  - Gradient focusing: sigmoid gate suppresses gradients from safe obstacles

Notation
--------
    X_t  = [w1, ..., wT]  in R^{T x 2}   full trajectory
    w_i  = (x_i, y_i)                     single waypoint
    p_j  = (px_j, py_j, r_j)              obstacle centre + radius
    N                                      number of obstacles
    T                                      number of waypoints
    c                                      softplus sharpness parameter

Softplus distance transformation (eq. 24)
------------------------------------------
    d_ij   = (px_j - x_i)^2 + (py_j - y_i)^2 - r_j^2 + gamma_delta
    d~_ij  = -c * softplus(-d_ij/c) + log(2)
           = -c * log(1 + exp(-d_ij/c)) + log(2)

    d_ij >> 0 => d~_ij ≈ log(2)   (safe: bounded above)
    d_ij << 0 => d~_ij ≈ d_ij     (unsafe: passes through unchanged)
    d~_ij <= log(2) always

Numerically stabilized two-level softmin
------------------------------------------
    d~_min,i = min_j d~_ij
    h(w_i)   = d~_min,i - k1 * log(1/N * sum_j exp(-(d~_ij - d~_min,i)/k1))  (eq. 30)

    h_min    = min_i h(w_i)
    h(X_t)   = h_min - k2 * log(1/T * sum_i exp(-(h(w_i) - h_min)/k2))       (eq. 27)

Stabilized softmax weights (eq. 28, 31)
-----------------------------------------
    sigma_i  = exp(-(h(w_i) - h_min)/k2)  / Z_k2      waypoint weights
    alpha_ij = exp(-(d~_ij - d~_min,i)/k1) / Z_k1_i   obstacle weights

Gradient with sigmoid gate (eq. 35-36)
----------------------------------------
    d d~_ij / d w_i     = -2 * sigmoid(-d_ij/c) * (p_j - w_i)      (eq. 34)
    nabla_wi h(w_i)     = -2 * sum_j alpha_ij * sigmoid(-d_ij/c) * (p_j - w_i)  (eq. 35)
    nabla_wi h(X_t)     = sigma_i * nabla_wi h(w_i)                 (eq. 29)
    nabla_{X_t} h(X_t)  = {nabla_wi h(X_t)}_{i=1}^T                 (eq. 36)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Raw signed distance  d_ij  (eq. 23)
# ---------------------------------------------------------------------------

def signed_distance_circle(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    d_ij = (px_j - x_i)^2 + (py_j - y_i)^2 - r_j^2 + gamma_delta

    Positive => waypoint is outside obstacle j (safe)
    Negative => waypoint is inside  obstacle j (unsafe)

    Args:
        wi          : [2]    waypoint (x_i, y_i)
        obstacles   : [N, 3] each row is (px_j, py_j, r_j)
        gamma_delta : safety margin γδ

    Returns:
        d : [N]
    """
    px = obstacles[:, 0]
    py = obstacles[:, 1]
    r  = obstacles[:, 2]
    return (px - wi[0]) ** 2 + (py - wi[1]) ** 2 - r ** 2 + gamma_delta


# ---------------------------------------------------------------------------
# 2. Softplus distance transformation  d~_ij  (eq. 24)
# ---------------------------------------------------------------------------

def softplus_distance(
    d: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """
    d~_ij = -c * softplus(-d_ij / c) + log(2)
          = -c * log(1 + exp(-d_ij / c)) + log(2)

    Accepts any tensor shape.

    Args:
        d : raw signed distances (any shape)
        c : sharpness parameter (c > 0)

    Returns:
        d_tilde : same shape as d, bounded above by log(2)
    """
    return -c * F.softplus(-d / c) + math.log(2)


# ---------------------------------------------------------------------------
# 3. Waypoint-level CBF  h(w_i)  (eq. 30)  — numerically stabilised
# ---------------------------------------------------------------------------

def waypoint_cbf(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Stabilised softmin of d~_ij over obstacles for a single waypoint.

        d~_min,i = min_j d~_ij
        h(w_i)   = d~_min,i - k1 * log(1/N * sum_j exp(-(d~_ij - d~_min,i)/k1))

    Args:
        wi          : [2]
        obstacles   : [N, 3]
        k1          : softmin temperature (obstacle level)
        gamma_delta : distance margin γδ
        c           : softplus sharpness

    Returns:
        h_wi : scalar tensor
    """
    d       = signed_distance_circle(wi, obstacles, gamma_delta)  # [N]
    d_tilde = softplus_distance(d, c)                              # [N]
    N       = d_tilde.shape[0]
    d_tilde_min = d_tilde.min()
    shifted = d_tilde - d_tilde_min                               # [N], >= 0
    return d_tilde_min - k1 * torch.logsumexp(-shifted / k1, dim=0) + k1 * math.log(N)


# ---------------------------------------------------------------------------
# 4. Trajectory-level CBF  h(X_t)  (eq. 27)  — numerically stabilised
# ---------------------------------------------------------------------------

def trajectory_cbf(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Stabilised softmin of h(w_i) over waypoints.

        h(X_t) = h_min - k2 * log(1/T * sum_i exp(-(h(w_i) - h_min)/k2))

    Fully vectorised: all d_ij as [T, N] in one broadcast.

    Args:
        Xt          : [T, 2]
        obstacles   : [N, 3]
        k1          : softmin temperature (obstacle level)
        k2          : softmin temperature (waypoint level)
        gamma_delta : distance margin γδ
        c           : softplus sharpness

    Returns:
        h_Xt : scalar tensor
    """
    centres = obstacles[:, :2]  # [N, 2]
    radii   = obstacles[:, 2]   # [N]

    diff  = Xt.unsqueeze(1) - centres.unsqueeze(0)                           # [T, N, 2]
    d_all = (diff ** 2).sum(dim=-1) - radii.unsqueeze(0) ** 2 + gamma_delta  # [T, N]
    d_tilde = softplus_distance(d_all, c)                                     # [T, N]

    N, T = d_all.shape[1], d_all.shape[0]
    d_tilde_min = d_tilde.min(dim=1, keepdim=True).values                    # [T, 1]
    shifted_d   = d_tilde - d_tilde_min                                      # [T, N], >= 0
    h_wi = (
        d_tilde_min.squeeze(1)
        - k1 * torch.logsumexp(-shifted_d / k1, dim=1)
        + k1 * math.log(N)
    )  # [T]

    h_min     = h_wi.min()
    shifted_h = h_wi - h_min                                                 # [T], >= 0
    return h_min - k2 * torch.logsumexp(-shifted_h / k2, dim=0) + k2 * math.log(T)


# ---------------------------------------------------------------------------
# 5. Gradient  d d_ij / d w_i  (eq. 29 original — unchanged)
# ---------------------------------------------------------------------------

def grad_dij_dwi(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
) -> torch.Tensor:
    """
    d d_ij / d w_i = 2 (w_i - p_j)

    Args:
        wi        : [2]
        obstacles : [N, 3]

    Returns:
        grad : [N, 2]
    """
    pj = obstacles[:, :2]
    return 2.0 * (wi.unsqueeze(0) - pj)


# ---------------------------------------------------------------------------
# 6. Gradient  nabla_wi h(w_i)  — softplus formulation (eq. 35)
# ---------------------------------------------------------------------------

def grad_hwi_dwi(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    nabla_wi h(w_i) = -2 * sum_j alpha_ij * sigmoid(-d_ij/c) * (p_j - w_i)

    where alpha_ij = stabilised softmax(-d~_ij/k1)  (obstacle weights, eq. 31).
    The sigmoid factor is the soft gate: kills gradients from safe obstacles.

    Args:
        wi          : [2]
        obstacles   : [N, 3]
        k1          : obstacle softmin temperature
        gamma_delta : distance margin
        c           : softplus sharpness

    Returns:
        grad : [2]
    """
    d       = signed_distance_circle(wi, obstacles, gamma_delta)  # [N]
    d_tilde = softplus_distance(d, c)                              # [N]

    d_tilde_min = d_tilde.min()
    alpha_ij    = F.softmax(-(d_tilde - d_tilde_min) / k1, dim=0)  # [N]

    gate = torch.sigmoid(-d / c)  # [N] — sigmoid gate

    pj           = obstacles[:, :2]          # [N, 2]
    displacement = pj - wi.unsqueeze(0)      # [N, 2]  (p_j - w_i)

    return -2.0 * ((alpha_ij * gate).unsqueeze(-1) * displacement).sum(dim=0)  # [2]


# ---------------------------------------------------------------------------
# 7. Gradient  nabla_wi h(X_t)  (eq. 29 — vectorised)
# ---------------------------------------------------------------------------

def grad_hXt_dwi(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    nabla_wi h(X_t) = sigma_i * nabla_wi h(w_i)   for all i simultaneously.

    This is identical to grad_hXt_dXt — the per-waypoint blocks assembled.
    """
    return grad_hXt_dXt(Xt, obstacles, k1, k2, gamma_delta, c)


# ---------------------------------------------------------------------------
# 8. Full gradient  nabla_{X_t} h(X_t)  (eq. 36)
# ---------------------------------------------------------------------------

def grad_hXt_dXt(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Full gradient of h(X_t) w.r.t. the trajectory X_t (eq. 36):

        nabla_{X_t} h(X_t) = { -2 sigma_i sum_j alpha_ij sigmoid(-d_ij/c) (p_j - w_i) }_{i=1}^T

    Shape is [T, 2], matching the layout of X_t.

    When no obstacles are present, returns a zero tensor of shape [T, 2].

    Args:
        Xt          : [T, 2] full trajectory
        obstacles   : [N, 3] obstacle tensor (px, py, r) — N >= 1
        k1          : softmin temperature (obstacle level)
        k2          : softmin temperature (waypoint level)
        gamma_delta : distance margin γδ
        c           : softplus sharpness

    Returns:
        grad : [T, 2]

    Example::
        >>> Xt = torch.randn(10, 2)
        >>> obs = torch.tensor([[1.0, 0.0, 0.5]])
        >>> g = grad_hXt_dXt(Xt, obs, k1=1.0, k2=1.0)
        >>> g.shape
        torch.Size([10, 2])
    """
    if obstacles.shape[0] == 0:
        return torch.zeros_like(Xt)

    centres = obstacles[:, :2]  # [N, 2]
    radii   = obstacles[:, 2]   # [N]

    # Raw signed distances [T, N]
    diff  = Xt.unsqueeze(1) - centres.unsqueeze(0)                           # [T, N, 2]
    d_all = (diff ** 2).sum(dim=-1) - radii.unsqueeze(0) ** 2 + gamma_delta  # [T, N]

    # Softplus transformed distances [T, N]
    d_tilde = softplus_distance(d_all, c)

    # Stabilised waypoint-level CBF values [T]
    N, T = d_all.shape[1], d_all.shape[0]
    d_tilde_min = d_tilde.min(dim=1, keepdim=True).values  # [T, 1]
    shifted_d   = d_tilde - d_tilde_min                    # [T, N]
    h_wi = (
        d_tilde_min.squeeze(1)
        - k1 * torch.logsumexp(-shifted_d / k1, dim=1)
        + k1 * math.log(N)
    )  # [T]

    # Obstacle weights alpha_ij [T, N]
    alpha_ij = F.softmax(-shifted_d / k1, dim=1)

    # Waypoint weights sigma_i [T]
    h_min     = h_wi.min()
    shifted_h = h_wi - h_min
    sigma_i   = F.softmax(-shifted_h / k2, dim=0)

    # Sigmoid gate sigma(-d_ij/c) [T, N]
    gate = torch.sigmoid(-d_all / c)

    # Inner gradient: -2 sum_j alpha_ij * gate * (p_j - w_i)  [T, 2]
    disp    = centres.unsqueeze(0) - Xt.unsqueeze(1)        # [T, N, 2]  p_j - w_i
    weights = (alpha_ij * gate).unsqueeze(-1)               # [T, N, 1]
    grad_hi = -2.0 * (weights * disp).sum(dim=1)            # [T, 2]

    return sigma_i.unsqueeze(-1) * grad_hi                   # [T, 2]


# ---------------------------------------------------------------------------
# 9. Full CBF metrics for the visualiser step inspector
# ---------------------------------------------------------------------------

def compute_cbf_metrics(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    c: float,
    k1: float,
    k2: float,
    gamma_delta: float,
) -> dict:
    """
    Compute every CBF quantity needed by the step inspector.

    Args:
        Xt          : [T, 2] trajectory
        obstacles   : [N, 3] (px, py, r)
        c           : softplus sharpness
        k1          : obstacle softmin temperature
        k2          : waypoint softmin temperature
        gamma_delta : safety margin

    Returns dict with:
        h_Xt    : float          — scalar trajectory-level CBF value
        d_raw   : list[float, T] — min raw signed distance per waypoint
        d_tilde : list[float, T] — softplus of that min distance
        h_wi    : list[float, T] — waypoint-level CBF values
        sigma_i : list[float, T] — waypoint softmax weights (sum to 1.0)
        grad_x  : list[float, T] — x-component of nabla_{X_t} h(X_t)
        grad_y  : list[float, T] — y-component
    """
    if obstacles.shape[0] == 0:
        T = Xt.shape[0]
        return {
            'h_Xt':    0.0,
            'd_raw':   [0.0] * T,
            'd_tilde': [0.0] * T,
            'h_wi':    [0.0] * T,
            'sigma_i': [1.0 / T] * T,
            'grad_x':  [0.0] * T,
            'grad_y':  [0.0] * T,
        }

    centres = obstacles[:, :2]
    radii   = obstacles[:, 2]
    N = obstacles.shape[0]
    T = Xt.shape[0]

    # Raw distances [T, N]
    diff  = Xt.unsqueeze(1) - centres.unsqueeze(0)
    d_all = (diff ** 2).sum(dim=-1) - radii.unsqueeze(0) ** 2 + gamma_delta

    # Softplus distances [T, N]
    d_tilde_all = softplus_distance(d_all, c)

    # Stabilised waypoint-level CBF [T]
    d_tilde_min = d_tilde_all.min(dim=1, keepdim=True).values  # [T, 1]
    shifted_d   = d_tilde_all - d_tilde_min                    # [T, N]
    h_wi = (
        d_tilde_min.squeeze(1)
        - k1 * torch.logsumexp(-shifted_d / k1, dim=1)
        + k1 * math.log(N)
    )  # [T]

    # Waypoint weights sigma_i [T]
    h_min     = h_wi.min()
    shifted_h = h_wi - h_min
    sigma_i   = F.softmax(-shifted_h / k2, dim=0)

    # Trajectory-level CBF (scalar)
    h_Xt = (
        h_min - k2 * torch.logsumexp(-shifted_h / k2, dim=0) + k2 * math.log(T)
    ).item()

    # Obstacle weights alpha_ij [T, N]
    alpha_ij = F.softmax(-shifted_d / k1, dim=1)

    # Sigmoid gate [T, N]
    gate = torch.sigmoid(-d_all / c)

    # Gradient [T, 2]
    disp    = centres.unsqueeze(0) - Xt.unsqueeze(1)
    weights = (alpha_ij * gate).unsqueeze(-1)
    grad_hi = -2.0 * (weights * disp).sum(dim=1)
    grad    = sigma_i.unsqueeze(-1) * grad_hi  # [T, 2]

    # Per-waypoint display value: the most dangerous (minimum raw distance) obstacle
    min_obs_idx  = d_all.argmin(dim=1)          # [T]
    arange_T     = torch.arange(T, device=Xt.device)
    d_raw_out    = d_all[arange_T, min_obs_idx]       # [T]
    d_tilde_out  = d_tilde_all[arange_T, min_obs_idx] # [T]

    return {
        'h_Xt':    h_Xt,
        'd_raw':   d_raw_out.tolist(),
        'd_tilde': d_tilde_out.tolist(),
        'h_wi':    h_wi.tolist(),
        'sigma_i': sigma_i.tolist(),
        'grad_x':  grad[:, 0].tolist(),
        'grad_y':  grad[:, 1].tolist(),
    }
