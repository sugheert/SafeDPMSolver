"""
VEDiffusion: Variance-Exploding SDE training wrapper (SMLD / NCSN-style).

Noise schedule: geometric interpolation between sigma_min and sigma_max
    sigma_i = sigma_min * (sigma_max / sigma_min)^(i / N)   for i = 0 ... N

Training loss (denoising score matching, predicting noise eps):
    L = E_{i, x0, eps} [ || eps_theta(x0 + sigma_i * eps, sigma_i, x_start, x_goal) - eps ||^2 ]

x0 is a full trajectory [B, T, 2]; sigma is broadcast over T and 2.
"""

import math
import torch
import torch.nn as nn


class VEDiffusion(nn.Module):
    """
    VE-SDE training wrapper.

    Args:
        model      : ScoreNet instance (or any callable with the same signature)
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

        # Precompute the sigma schedule as a registered buffer [N+1]
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
        x0:      torch.Tensor,   # [B, T, 2]  clean trajectories
        x_start: torch.Tensor,   # [B, 2]      trajectory starts
        x_goal:  torch.Tensor,   # [B, 2]      trajectory goals
    ):
        """
        Compute DSM loss.  Returns (loss_scalar, info_dict).
        """
        B      = x0.shape[0]
        device = x0.device

        # Sample random noise levels
        idx   = torch.randint(0, self.n_levels, (B,), device=device)
        sigma = self.sigmas[idx]                        # [B]

        # Broadcast sigma over trajectory dimensions
        eps     = torch.randn_like(x0)                  # [B, T, 2]
        x_noisy = x0 + sigma[:, None, None] * eps       # [B, T, 2]

        # Predict noise over the full trajectory
        eps_pred = self.model(x_noisy, sigma, x_start, x_goal)  # [B, T, 2]

        loss = ((eps_pred - eps) ** 2).mean()
        info = {
            'loss':       loss.item(),
            'sigma_mean': sigma.mean().item(),
        }
        return loss, info

    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
