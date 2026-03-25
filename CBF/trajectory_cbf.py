"""
Trajectory-level Control Barrier Functions for circular obstacles.

Reference: FullDerv.pdf, Section 4 — "The CBF of X_t"

Notation
--------
    X_t  = [w1, ..., wT]  in R^{T x 2}   full trajectory
    w_i  = (x_i, y_i)                     single waypoint
    p_j  = (px_j, py_j, r_j)              obstacle centre + radius
    N                                      number of obstacles
    T                                      number of waypoints

Two-level softmin structure
---------------------------
    d_ij   = (px_j - x_i)^2 + (py_j - y_i)^2 - r_j^2 + gamma_delta        (eq. 23)
    h(w_i) = -k1 log( (1/N) sum_j exp(-d_ij / k1) )                        (eq. 24)
    h(X_t) = -k2 log( (1/T) sum_i exp(-h(w_i) / k2) )                      (eq. 25)

Softmax weights
---------------
    m1_ij = softmax(-d_ij / k1)   over obstacles   (eq. 28)
    m2_i  = softmax(-h(w_i) / k2) over waypoints   (eq. 26)

Gradients (analytic)
--------------------
    d d_ij  / d w_i    = -2 (p_j - w_i)                              (eq. 29)
    d h(wi) / d w_i    = -2 sum_j m1_ij (p_j - w_i)                  (eq. 30)
    d h(Xt) / d w_i    = m2_i * grad_{w_i} h(w_i)                    (eq. 27)
    grad_{X_t} h(X_t)  = {-2 m2_i sum_j m1_ij (p_j - w_i)}_{i}      (eq. 31)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Signed distance from a single waypoint to each circular obstacle (eq. 23)
# ---------------------------------------------------------------------------

def signed_distance_circle(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Squared-distance-based signed distance from waypoint w_i to each obstacle.

        d_ij = (px_j - x_i)^2 + (py_j - y_i)^2 - r_j^2 + gamma_delta

    Positive  => waypoint is outside obstacle j  (safe)
    Negative  => waypoint is inside  obstacle j  (unsafe)

    Args:
        wi          : [2]    waypoint (x_i, y_i)
        obstacles   : [N, 3] each row is (px_j, py_j, r_j)
        gamma_delta : scalar margin term γδ (default 0.0)

    Returns:
        d : [N]
    """
    px = obstacles[:, 0]   # [N]
    py = obstacles[:, 1]   # [N]
    r  = obstacles[:, 2]   # [N]
    return (px - wi[0]) ** 2 + (py - wi[1]) ** 2 - r ** 2 + gamma_delta


# ---------------------------------------------------------------------------
# 2. Waypoint-level CBF  (eq. 24)
# ---------------------------------------------------------------------------

def waypoint_cbf(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Normalized softmin of d_ij over obstacles for a single waypoint.

        h(w_i) = -k1 * log( (1/N) * sum_j exp(-d_ij / k1) )

    Args:
        wi          : [2]    waypoint
        obstacles   : [N, 3] obstacle tensor (px, py, r)
        k1          : softmin temperature (obstacle level)
        gamma_delta : distance margin γδ

    Returns:
        h_wi : scalar tensor
    """
    d = signed_distance_circle(wi, obstacles, gamma_delta)   # [N]
    N = d.shape[0]
    return -k1 * torch.logsumexp(-d / k1, dim=0) + k1 * math.log(N)


# ---------------------------------------------------------------------------
# 3. Trajectory-level CBF  (eq. 25)
# ---------------------------------------------------------------------------

def trajectory_cbf(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Normalized softmin of h(w_i) over waypoints for the full trajectory.

        h(X_t) = -k2 * log( (1/T) * sum_i exp(-h(w_i) / k2) )

    Fully vectorised: computes all d_ij as [T, N] in a single broadcast op.

    Args:
        Xt          : [T, 2] full trajectory
        obstacles   : [N, 3] obstacle tensor (px, py, r)
        k1          : softmin temperature (obstacle level)
        k2          : softmin temperature (waypoint / trajectory level)
        gamma_delta : distance margin γδ

    Returns:
        h_Xt : scalar tensor
    """
    # d_all[i, j] = d_ij
    centres = obstacles[:, :2]   # [N, 2]
    radii   = obstacles[:, 2]    # [N]

    diff  = Xt.unsqueeze(1) - centres.unsqueeze(0)          # [T, N, 2]
    d_all = (diff ** 2).sum(dim=-1) - radii.unsqueeze(0) ** 2 + gamma_delta  # [T, N]

    N, T = d_all.shape[1], d_all.shape[0]
    h_wi = -k1 * torch.logsumexp(-d_all / k1, dim=1) + k1 * math.log(N)   # [T]
    return -k2 * torch.logsumexp(-h_wi / k2, dim=0) + k2 * math.log(T)    # scalar


# ---------------------------------------------------------------------------
# 4. Gradient  d d_ij / d w_i  (eq. 29)
# ---------------------------------------------------------------------------

def grad_dij_dwi(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
) -> torch.Tensor:
    """
    Analytic gradient of d_ij w.r.t. the waypoint w_i.

        d d_ij / d w_i = -2 (p_j - w_i) = 2 (w_i - p_j)

    Args:
        wi        : [2]    waypoint
        obstacles : [N, 3] obstacle tensor (px, py, r)

    Returns:
        grad : [N, 2]  — one row per obstacle
    """
    pj = obstacles[:, :2]                    # [N, 2]
    return 2.0 * (wi.unsqueeze(0) - pj)     # [N, 2]


# ---------------------------------------------------------------------------
# 5. Gradient  d h(w_i) / d w_i  (eq. 30)
# ---------------------------------------------------------------------------

def grad_hwi_dwi(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Analytic gradient of h(w_i) w.r.t. waypoint w_i.

        m1_ij           = softmax(-d_ij / k1)           (obstacle weights, eq. 28)
        grad h(w_i)     = -2 sum_j m1_ij (p_j - w_i)   (eq. 30)

    Args:
        wi          : [2]    waypoint
        obstacles   : [N, 3] obstacle tensor
        k1          : softmin temperature (obstacle level)
        gamma_delta : distance margin γδ

    Returns:
        grad : [2]
    """
    d  = signed_distance_circle(wi, obstacles, gamma_delta)   # [N]
    m1 = F.softmax(-d / k1, dim=0)                            # [N]

    pj           = obstacles[:, :2]                            # [N, 2]
    displacement = pj - wi.unsqueeze(0)                        # [N, 2]  (p_j - w_i)

    return -2.0 * (m1.unsqueeze(-1) * displacement).sum(dim=0)   # [2]


# ---------------------------------------------------------------------------
# 6. Gradient  d h(X_t) / d w_i  (eq. 27)
# ---------------------------------------------------------------------------

def grad_hXt_dwi(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Gradient of the trajectory-level CBF h(X_t) w.r.t. each waypoint w_i.

        m2_i             = softmax(-h(w_i) / k2)   (waypoint weights, eq. 26)
        d h(X_t) / d w_i = m2_i * grad_{w_i} h(w_i)               (eq. 27)

    Fully vectorised over T waypoints.

    Args:
        Xt          : [T, 2] full trajectory
        obstacles   : [N, 3] obstacle tensor
        k1          : softmin temperature (obstacle level)
        k2          : softmin temperature (waypoint / trajectory level)
        gamma_delta : distance margin γδ

    Returns:
        grads : [T, 2]  — gradient block for every waypoint
    """
    centres = obstacles[:, :2]   # [N, 2]
    radii   = obstacles[:, 2]    # [N]

    # --- distances  [T, N] ---------------------------------------------------
    diff  = Xt.unsqueeze(1) - centres.unsqueeze(0)                # [T, N, 2]
    d_all = (diff ** 2).sum(dim=-1) - radii.unsqueeze(0) ** 2 + gamma_delta   # [T, N]

    # --- waypoint-level CBF values and weights -------------------------------
    h_wi = -k1 * torch.logsumexp(-d_all / k1, dim=1)             # [T]
    m2   = F.softmax(-h_wi / k2, dim=0)                           # [T]  waypoint weights

    # --- obstacle weights for each waypoint ----------------------------------
    m1 = F.softmax(-d_all / k1, dim=1)                            # [T, N]  obstacle weights

    # --- inner gradient: -2 sum_j m1_ij (p_j - w_i) -------------------------
    # disp[i, j] = p_j - w_i   shape [T, N, 2]
    disp    = centres.unsqueeze(0) - Xt.unsqueeze(1)
    grad_hi = -2.0 * (m1.unsqueeze(-1) * disp).sum(dim=1)         # [T, 2]

    # --- scale by m2_i  ------------------------------------------------------
    return m2.unsqueeze(-1) * grad_hi                              # [T, 2]


# ---------------------------------------------------------------------------
# 7. Overall gradient  d h(X_t) / d X_t  (eq. 31)
# ---------------------------------------------------------------------------

def grad_hXt_dXt(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    """
    Full gradient of h(X_t) w.r.t. the trajectory X_t, assembled blockwise.

        grad_{X_t} h(X_t) = { -2 m2_i sum_j m1_ij (p_xj - x_i, p_yj - y_i) }_{i=1}^T

    Shape is [T, 2], matching the layout of X_t. This is the quantity used
    directly in the DPM-Solver CBF correction step.

    When no obstacles are present, returns a zero tensor of shape [T, 2].

    Args:
        Xt          : [T, 2] full trajectory
        obstacles   : [N, 3] obstacle tensor (px, py, r) — N >= 1
        k1          : softmin temperature (obstacle level);  smaller => harder min
        k2          : softmin temperature (waypoint level);  smaller => harder min
        gamma_delta : distance margin γδ (adds a safety buffer)

    Returns:
        grad : [T, 2]

    Example::
        >>> Xt = torch.randn(10, 2)
        >>> obs = torch.tensor([[1.0, 0.0, 0.5]])   # one obstacle
        >>> g = grad_hXt_dXt(Xt, obs, k1=1.0, k2=1.0)
        >>> g.shape
        torch.Size([10, 2])
    """
    if obstacles.shape[0] == 0:
        return torch.zeros_like(Xt)

    # This is identical to grad_hXt_dwi — the full assembled gradient is
    # exactly the per-waypoint blocks stacked along the first dimension.
    return grad_hXt_dwi(Xt, obstacles, k1, k2, gamma_delta)
