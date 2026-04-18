"""
ve_diffusion_ellipsoids_rasterizedlocalfeats.py — VEDiffusion for local-feature-sampling model.

Identical interface to ve_diffusion_ellipsoids.py:
    - Accepts occ_map [B, 1, H, W] in loss()
    - CFG dropout: replaces occ_map with zeros for p_uncond fraction of the batch
    - Calls self.model(x_noisy, sigma, x_start, x_goal, occ_map_dropped)

The underlying TemporalUnet (score_net_ellipsoids_rasterizedlocalfeats.py) handles
local feature sampling internally, so no changes are needed to the diffusion wrapper.
"""

import math
import torch
import torch.nn as nn


class VEDiffusion(nn.Module):
    """
    VE-SDE training wrapper with occupancy-map conditioning and CFG dropout.

    Args:
        model      : TemporalUnet instance from score_net_ellipsoids_rasterizedlocalfeats.py
        sigma_min  : minimum noise level
        sigma_max  : maximum noise level
        n_levels   : number of discrete noise levels (N)
    """

    def __init__(
        self,
        model,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        n_levels:  int   = 1000,
    ):
        super().__init__()
        self.model     = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_levels  = n_levels

        levels = torch.arange(n_levels + 1, dtype=torch.float32)
        sigmas = sigma_min * (sigma_max / sigma_min) ** (levels / n_levels)
        self.register_buffer('sigmas', sigmas)

        self._log_ratio = math.log(sigma_max / sigma_min)

    def sigma_dot(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self._log_ratio

    def loss(
        self,
        x0:       torch.Tensor,   # [B, T, 2]     clean trajectories
        x_start:  torch.Tensor,   # [B, 2]         trajectory starts
        x_goal:   torch.Tensor,   # [B, 2]         trajectory goals
        occ_map:  torch.Tensor,   # [B, 1, H, W]   rasterized occupancy map
        p_uncond: float = 0.1,
    ):
        """
        Compute DSM loss with CFG dropout.

        With probability p_uncond per sample, occ_map is replaced with zeros,
        teaching both conditional and unconditional distributions in one model.

        Returns (loss_scalar, info_dict).
        """
        B      = x0.shape[0]
        device = x0.device

        idx   = torch.randint(0, self.n_levels, (B,), device=device)
        sigma = self.sigmas[idx]

        eps     = torch.randn_like(x0)
        x_noisy = x0 + sigma[:, None, None] * eps

        if p_uncond > 0.0:
            drop_mask = (torch.rand(B, device=device) < p_uncond)
            occ_map_dropped = occ_map.clone()
            occ_map_dropped[drop_mask] = 0.0
        else:
            occ_map_dropped = occ_map

        eps_pred = self.model(x_noisy, sigma, x_start, x_goal, occ_map_dropped)

        loss = ((eps_pred - eps) ** 2).mean()
        info = {
            'loss':       loss.item(),
            'sigma_mean': sigma.mean().item(),
        }
        return loss, info

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
