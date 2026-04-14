"""
ve_diffusion_ellipsoids.py — VEDiffusion with ellipsoid obstacle conditioning + CFG.

Extends models/ve_diffusion.py to accept ellipsoid obstacle information
([B, 5, 4]) and support Classifier-Free Guidance (CFG) training via
conditional dropout.

CFG training:
    With probability p_uncond per sample, the ellipsoid conditioning is
    replaced with zeros before the forward pass. This teaches the network
    both the conditional p(trajectory | start, goal, ellipsoids) and the
    unconditional p(trajectory | start, goal) distributions in one model.

Noise schedule: geometric interpolation between sigma_min and sigma_max
    sigma_i = sigma_min * (sigma_max / sigma_min)^(i / N)   for i = 0 ... N

Training loss (denoising score matching, predicting noise eps):
    L = E_{i, x0, eps} [ || eps_theta(x0 + sigma_i * eps, sigma_i,
                                       x_start, x_goal, ellipsoids_dropped)
                           - eps ||^2 ]
"""

import math
import torch
import torch.nn as nn


class VEDiffusion(nn.Module):
    """
    VE-SDE training wrapper with ellipsoid conditioning and CFG dropout.

    Args:
        model      : TemporalUnet instance from score_net_ellipsoids.py
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

    # ------------------------------------------------------------------

    def sigma_dot(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self._log_ratio

    # ------------------------------------------------------------------

    def loss(
        self,
        x0:         torch.Tensor,   # [B, T, 2]   clean trajectories
        x_start:    torch.Tensor,   # [B, 2]       trajectory starts
        x_goal:     torch.Tensor,   # [B, 2]       trajectory goals
        ellipsoids: torch.Tensor,   # [B, 5, 4]    obstacle set (cx,cy,a,b)
        p_uncond:   float = 0.1,    # probability of dropping ellipsoid conditioning
    ):
        """
        Compute DSM loss with CFG dropout.

        For each sample in the batch, ellipsoid conditioning is independently
        zeroed out with probability p_uncond, enabling CFG at inference time.

        Returns (loss_scalar, info_dict).
        """
        B      = x0.shape[0]
        device = x0.device

        # Sample random noise levels
        idx   = torch.randint(0, self.n_levels, (B,), device=device)
        sigma = self.sigmas[idx]                        # [B]

        # Add noise
        eps     = torch.randn_like(x0)                  # [B, T, 2]
        x_noisy = x0 + sigma[:, None, None] * eps       # [B, T, 2]

        # CFG dropout: zero out ellipsoid conditioning per sample
        if p_uncond > 0.0:
            drop_mask = (torch.rand(B, device=device) < p_uncond)  # [B]
            ellipsoids_dropped = ellipsoids.clone()
            ellipsoids_dropped[drop_mask] = 0.0
        else:
            ellipsoids_dropped = ellipsoids

        # Predict noise
        eps_pred = self.model(x_noisy, sigma, x_start, x_goal, ellipsoids_dropped)

        loss = ((eps_pred - eps) ** 2).mean()
        info = {
            'loss':       loss.item(),
            'sigma_mean': sigma.mean().item(),
        }
        return loss, info

    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
