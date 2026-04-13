"""
Trajectory-level Control Barrier Functions for degree-4 hyperellipsoid obstacles.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

EPSILON = 1e-6

def signed_distance_hyperellipsoid(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    gamma_delta: float = 0.0,
) -> torch.Tensor:
    px = obstacles[:, 0]
    py = obstacles[:, 1]
    a  = obstacles[:, 2]
    b  = obstacles[:, 3]
    term_x = ((px - wi[0]) / a) ** 4
    term_y = ((py - wi[1]) / b) ** 4
    rho = (term_x + term_y + EPSILON) ** 0.25
    return rho - 1.0 + gamma_delta

def softplus_distance(d: torch.Tensor, c: float) -> torch.Tensor:
    return -c * F.softplus(-d / c) + c*math.log(2)

def waypoint_cbf(
    wi: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
) -> torch.Tensor:
    d = signed_distance_hyperellipsoid(wi, obstacles, gamma_delta)
    d_tilde = softplus_distance(d, c)
    N = d_tilde.shape[0]
    d_tilde_min = d_tilde.min()
    shifted = d_tilde - d_tilde_min
    return d_tilde_min - k1 * torch.logsumexp(-shifted / k1, dim=0) #+ k1 * math.log(N)

def trajectory_cbf(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
    use_softplus: bool = True,
) -> torch.Tensor:
    centres = obstacles[:, :2]  # [N, 2]
    a_b   = obstacles[:, 2:4]   # [N, 2]

    diff  = centres.unsqueeze(0) - Xt.unsqueeze(1)                           # [T, N, 2]
    term4 = (diff / a_b.unsqueeze(0)) ** 4 # [T, N, 2]
    rho = (term4.sum(dim=-1) + EPSILON) ** 0.25 # [T, N]
    d_all = rho - 1.0 + gamma_delta  # [T, N]
    d_tilde = softplus_distance(d_all, c) if use_softplus else d_all          # [T, N]

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
    return h_min - k2 * torch.logsumexp(-shifted_h / k2, dim=0) #+ k2 * math.log(T)

def grad_hXt_dXt(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    k1: float,
    k2: float,
    gamma_delta: float = 0.0,
    c: float = 1.0,
    use_softplus: bool = True,
) -> torch.Tensor:
    if obstacles.shape[0] == 0:
        return torch.zeros_like(Xt)
        
    centres = obstacles[:, :2]  # [N, 2]
    a_b   = obstacles[:, 2:4]   # [N, 2]

    diff  = centres.unsqueeze(0) - Xt.unsqueeze(1)                           # [T, N, 2]
    term4 = (diff / a_b.unsqueeze(0)) ** 4 # [T, N, 2]
    rho = (term4.sum(dim=-1) + EPSILON) ** 0.25 # [T, N]
    d_all = rho - 1.0 + gamma_delta  # [T, N]

    d_tilde = softplus_distance(d_all, c) if use_softplus else d_all

    N, T = d_all.shape[1], d_all.shape[0]
    d_tilde_min = d_tilde.min(dim=1, keepdim=True).values  # [T, 1]
    shifted_d   = d_tilde - d_tilde_min                    # [T, N]
    h_wi = (
        d_tilde_min.squeeze(1)
        - k1 * torch.logsumexp(-shifted_d / k1, dim=1)
        + k1 * math.log(N)
    )  # [T]

    alpha_ij = F.softmax(-shifted_d / k1, dim=1)

    h_min     = h_wi.min()
    shifted_h = h_wi - h_min
    sigma_i   = F.softmax(-shifted_h / k2, dim=0)

    gate = torch.sigmoid(-d_all / c) if use_softplus else torch.ones_like(d_all)

    # Gradient formulation: -1 * SUM[ alpha_ij * gate * (1 / rho^3) * ( (px - x)^3 / a^4 ) ]
    disp    = (diff ** 3) / (a_b.unsqueeze(0) ** 4)         # [T, N, 2]
    weights = (alpha_ij * gate / (rho ** 3)).unsqueeze(-1)  # [T, N, 1]
    grad_hi = -1.0 * (weights * disp).sum(dim=1)            # [T, 2]

    return sigma_i.unsqueeze(-1) * grad_hi                    # [T, 2]

def compute_cbf_metrics(
    Xt: torch.Tensor,
    obstacles: torch.Tensor,
    c: float,
    k1: float,
    k2: float,
    gamma_delta: float,
    use_softplus: bool = True,
) -> dict:
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
    a_b   = obstacles[:, 2:4]
    N = obstacles.shape[0]
    T = Xt.shape[0]

    diff  = centres.unsqueeze(0) - Xt.unsqueeze(1)                           # [T, N, 2]
    term4 = (diff / a_b.unsqueeze(0)) ** 4 # [T, N, 2]
    rho = (term4.sum(dim=-1) + EPSILON) ** 0.25 # [T, N]
    d_all = rho - 1.0 + gamma_delta  # [T, N]

    d_tilde_all = softplus_distance(d_all, c) if use_softplus else d_all

    d_tilde_min = d_tilde_all.min(dim=1, keepdim=True).values  # [T, 1]
    shifted_d   = d_tilde_all - d_tilde_min                    # [T, N]
    h_wi = (
        d_tilde_min.squeeze(1)
        - k1 * torch.logsumexp(-shifted_d / k1, dim=1)
        #+ k1 * math.log(N)
    )  # [T]

    h_min     = h_wi.min()
    shifted_h = h_wi - h_min
    sigma_i   = F.softmax(-shifted_h / k2, dim=0)

    h_Xt = (
        h_min - k2 * torch.logsumexp(-shifted_h / k2, dim=0) #+ k2 * math.log(T)
    ).item()

    alpha_ij = F.softmax(-shifted_d / k1, dim=1)
    gate = torch.sigmoid(-d_all / c) if use_softplus else torch.ones_like(d_all)

    disp    = (diff ** 3) / (a_b.unsqueeze(0) ** 4)         # [T, N, 2]
    weights = (alpha_ij * gate / (rho ** 3)).unsqueeze(-1)  # [T, N, 1]
    grad_hi = -1.0 * (weights * disp).sum(dim=1)
    grad    = sigma_i.unsqueeze(-1) * grad_hi  # [T, 2]

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
