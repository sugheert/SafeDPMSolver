"""
score_net_ellipsoids.py — TemporalUnet with local spatial conditioning for ellipsoidal obstacles.

Implements the "Lightweight Local PointNet" strategy from the implementation plan
(LocalSamplingImplementation.md). Replaces the global DeepSet FiLM encoder with a
trajectory-aligned LightweightLocalEncoder that performs early fusion:

    1. For every waypoint x_t, compute raw relative geometry to every obstacle.
    2. Pass through a shared MLP + max-pool (PointNet-style) → per-waypoint local vector l_t.
    3. Concatenate l_t to x_t BEFORE the U-Net downsampling path (early fusion).

Modified TemporalUnet.forward signature (unchanged from previous version):
    forward(x, sigma, x_start, x_goal, ellipsoids)
    where ellipsoids: [B, N, 4]  — each row (cx, cy, a, b); zeros = null/unconditional

Architecture:
    LightweightLocalEncoder: [B, T, 2] x [B, N, 4] -> [B, T, d_local]
        point_mlp : Linear(4 -> 128) + ReLU + Linear(128 -> 128) + ReLU   (per-pair)
        max-pool  : over N obstacles  -> [B, T, 128]
        proj_mlp  : Linear(128 -> 128) + ReLU + Linear(128 -> d_local)

    TemporalUnet first conv input: state_dim + d_local  (early fusion)
    cond_dim: time_emb_dim + conditioning_embed_dim  (no ellipsoid in global cond)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------

def group_norm_n_groups(n_channels, target_n_groups=8):
    if n_channels < target_n_groups:
        return 1
    for n_groups in range(target_n_groups, target_n_groups + 10):
        if n_channels % n_groups == 0:
            return n_groups
    return 1


# ---------------------------------------------------------------------------
# Time / sigma encoding (unchanged)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device   = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimeEncoder(nn.Module):
    """Encodes a per-batch scalar (sigma in VE-SDE) → [B, dim_out]."""
    def __init__(self, dim: int = 32, dim_out: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim_out),
        )

    def forward(self, x):   # x: [B]
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Conv building blocks (unchanged)
# ---------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish"""
    def __init__(self, inp_channels, out_channels, kernel_size=5, padding=None, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1,
                      padding=padding if padding is not None else kernel_size // 2),
            Rearrange('batch channels n -> batch channels 1 n'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 n -> batch channels n'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResidualTemporalBlock(nn.Module):
    """Two Conv1dBlocks with a conditioning injection and a residual conv."""
    def __init__(self, inp_channels, out_channels, cond_embed_dim, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size,
                        n_groups=group_norm_n_groups(out_channels)),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=group_norm_n_groups(out_channels)),
        ])
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_embed_dim, out_channels),
            Rearrange('b t -> b t 1'),
        )
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, kernel_size=1)
            if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, c):
        # x: [B, inp_ch, T]   c: [B, cond_dim]
        h = self.blocks[0](x) + self.cond_mlp(c)
        h = self.blocks[1](h)
        return h + self.residual_conv(x)


# ---------------------------------------------------------------------------
# Lightweight Local PointNet encoder  (replaces EllipsoidFiLMEncoder)
# ---------------------------------------------------------------------------

class LightweightLocalEncoder(nn.Module):
    """
    Computes per-waypoint local spatial features relative to a set of ellipsoids.

    For each trajectory waypoint x_t and each obstacle o_i = (cx, cy, a, b):
        f_{t,i} = [x_t - cx,  y_t - cy,  a,  b]   ∈ R^4

    Processed through a shared MLP (λ_θ), max-pooled over obstacles (N dim),
    then projected (γ_φ) to produce the local conditioning vector l_t:
        e_{t,i} = λ_θ(f_{t,i})
        z_t     = max_i(e_{t,i})
        l_t     = γ_φ(z_t)  ∈ R^{d_local}

    Args:
        d_hidden : hidden dimension of point_mlp (128)
        d_local  : output dimension per waypoint  (64)

    Forward:
        x         : [B, T, 2]    noisy trajectory waypoints
        obstacles : [B, N, 4]    ellipsoid params (cx, cy, a, b)
        returns   : [B, T, d_local]
    """
    def __init__(self, d_hidden: int = 128, d_local: int = 64):
        super().__init__()

        # λ_theta: Shared MLP for pairwise relative features
        self.point_mlp = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

        # γ_phi: Final projection after max-pooling
        self.proj_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_local),
        )

    def forward(self, x: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        _, N, _ = obstacles.shape

        if N == 0:
            d_local = self.proj_mlp[-1].out_features
            return torch.zeros(B, T, d_local, device=x.device)

        # 1. Expand for pairwise interactions: [B, T, N, ...]
        x_expanded   = x.unsqueeze(2).expand(B, T, N, 2)
        obs_expanded = obstacles.unsqueeze(1).expand(B, T, N, 4)

        dx = x_expanded[..., 0] - obs_expanded[..., 0]   # x - cx
        dy = x_expanded[..., 1] - obs_expanded[..., 1]   # y - cy
        a  = obs_expanded[..., 2]
        b  = obs_expanded[..., 3]

        # Raw relative features -> [B, T, N, 4]
        rel_features = torch.stack([dx, dy, a, b], dim=-1)

        # 2. Shared point MLP -> [B, T, N, d_hidden]
        e = self.point_mlp(rel_features)

        # 3. Symmetric max-pool over N -> [B, T, d_hidden]
        z = torch.max(e, dim=2)[0]

        # 4. Final projection -> [B, T, d_local]
        return self.proj_mlp(z)


# ---------------------------------------------------------------------------
# Temporal U-Net with local spatial conditioning (early fusion)
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """
    MPD-style 1D temporal U-Net for trajectory denoising with local spatial
    conditioning via early feature fusion (LightweightLocalEncoder).

    Early fusion: local features l_t ∈ R^{d_local} are concatenated to each
    waypoint x_t BEFORE the first convolution, so the U-Net sees
    [x_t || l_t] ∈ R^{state_dim + d_local} as its spatial input channel.

    Global conditioning (time + start/goal) is still injected at every
    ResidualTemporalBlock via the cond_dim vector, but ellipsoids are no longer
    part of the global context.

    Args:
        state_dim              : spatial dims per waypoint (2 for x,y)
        T_steps                : number of waypoints (trajectory horizon)
        unet_input_dim         : base channel width (32)
        dim_mults              : channel multipliers per U-Net level ((1,2,4))
        time_emb_dim           : output dim of the sigma (time) encoder (32)
        conditioning_embed_dim : dim to project cat(x_start, x_goal) into (4)
        local_hidden_dim       : hidden dim of LightweightLocalEncoder (128)
        local_dim              : per-waypoint output dim of local encoder (64)
    """

    def __init__(
        self,
        state_dim:              int   = 2,
        T_steps:                int   = 64,
        unet_input_dim:         int   = 32,
        dim_mults:              tuple = (1, 2, 4),
        time_emb_dim:           int   = 32,
        conditioning_embed_dim: int   = 4,
        local_hidden_dim:       int   = 128,
        local_dim:              int   = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.T_steps   = T_steps
        self.local_dim = local_dim

        # ---- Global conditioning (time + start/goal) ----
        self.context_proj = nn.Linear(state_dim * 2, conditioning_embed_dim)
        self.time_mlp     = TimeEncoder(dim=32, dim_out=time_emb_dim)

        # cond_dim fed to every ResidualTemporalBlock (no ellipsoid contribution)
        cond_dim = time_emb_dim + conditioning_embed_dim

        # ---- Local encoder (early fusion) ----
        self.local_encoder = LightweightLocalEncoder(
            d_hidden=local_hidden_dim,
            d_local=local_dim,
        )

        # ---- Channel dims: first level receives state_dim + local_dim ----
        fused_input_dim = state_dim + local_dim
        dims   = [fused_input_dim, *[unet_input_dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ---- Down path ----
        self.downs = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind == len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in,  dim_out, cond_dim),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # ---- Mid ----
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim)

        # ---- Up path ----
        self.ups = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim),
                ResidualTemporalBlock(dim_in,      dim_in, cond_dim),
                Upsample1d(dim_in),
            ]))

        # ---- Final projection ----
        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5,
                        n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, kernel_size=1),
        )

        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        x:          torch.Tensor,   # [B, T, 2]   noisy trajectory
        sigma:      torch.Tensor,   # [B]          noise level
        x_start:    torch.Tensor,   # [B, 2]       start condition
        x_goal:     torch.Tensor,   # [B, 2]       goal  condition
        ellipsoids: torch.Tensor,   # [B, N, 4]    ellipsoid set (cx,cy,a,b); zeros = null
    ) -> torch.Tensor:
        """Returns predicted noise eps : [B, T, 2]."""

        # 1. Global conditioning (time + start/goal)
        t_emb = self.time_mlp(sigma)                                    # [B, time_emb_dim]
        ctx   = self.context_proj(torch.cat([x_start, x_goal], dim=-1)) # [B, cond_emb_dim]
        c_emb = torch.cat([t_emb, ctx], dim=-1)                         # [B, cond_dim]

        # 2. Local conditioning — early fusion
        local_features = self.local_encoder(x, ellipsoids)              # [B, T, local_dim]
        x_fused = torch.cat([x, local_features], dim=-1)                # [B, T, state_dim+local_dim]

        # 3. Rearrange for 1D convolutions
        x_in = rearrange(x_fused, 'b t d -> b d t')                    # [B, state_dim+local_dim, T]

        skips = []
        for resnet1, resnet2, downsample in self.downs:
            x_in = resnet1(x_in, c_emb)
            x_in = resnet2(x_in, c_emb)
            skips.append(x_in)
            x_in = downsample(x_in)

        x_in = self.mid_block1(x_in, c_emb)
        x_in = self.mid_block2(x_in, c_emb)

        for resnet1, resnet2, upsample in self.ups:
            x_in = torch.cat([x_in, skips.pop()], dim=1)
            x_in = resnet1(x_in, c_emb)
            x_in = resnet2(x_in, c_emb)
            x_in = upsample(x_in)

        x_in = self.final_conv(x_in)                     # [B, 2, T]
        return rearrange(x_in, 'b d t -> b t d')         # [B, T, 2]


# Alias for backward-compat
ScoreNet = TemporalUnet
