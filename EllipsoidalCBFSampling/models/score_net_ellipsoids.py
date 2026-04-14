"""
score_net_ellipsoids.py — TemporalUnet conditioned on ellipsoidal obstacles.

Extends models/score_net.py with a DeepSet-based ellipsoid encoder that
encodes a set of up to 5 axis-aligned superellipsoid obstacles into a
256-dim conditioning vector, appended to the existing sigma + start/goal
conditioning.

Ellipsoid conditioning supports Classifier-Free Guidance (CFG): pass
zeros([B, 5, 4]) as the null (unconditional) ellipsoid input.

Architecture additions:
    EllipsoidFiLMEncoder: [B, 5, 4] -> [B, 256]
        embed  : Linear(4 -> 128) + ReLU
        layer1 : DeepSetLayer(128)      (equivariant)
        layer2 : DeepSetLayer(128)      (equivariant)
        rho    : max-pool + Linear(128->128) + ReLU + Linear(128->256)   (invariant)

Modified TemporalUnet.forward signature:
    forward(x, sigma, x_start, x_goal, ellipsoids)
    where ellipsoids: [B, 5, 4]  — each row (cx, cy, a, b); pad missing with zeros
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Helpers (unchanged from score_net.py)
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
# DeepSet ellipsoid encoder
# ---------------------------------------------------------------------------

class DeepSetLayer(nn.Module):
    """
    One equivariant DeepSet layer.

    Applies a local linear transform (lambda) and adds a global context term
    (gamma applied to the per-set max-pool), then activates with ReLU.

    Input/output shape: [B, N, dim]  (N = number of set elements)
    """
    def __init__(self, dim):
        super().__init__()
        self.lam = nn.Linear(dim, dim)   # local (element-wise) weight
        self.gam = nn.Linear(dim, dim)   # global (max-pool) weight

    def forward(self, x):
        # x: [B, N, dim]
        local_part  = self.lam(x)                          # [B, N, dim]
        global_max  = x.max(dim=1, keepdim=True)[0]        # [B, 1, dim]
        global_part = self.gam(global_max)                 # [B, 1, dim]  (broadcast)
        return F.relu(local_part + global_part)            # [B, N, dim]


class EllipsoidFiLMEncoder(nn.Module):
    """
    Encodes a set of ellipsoid obstacles into a single scene-level vector.

    Architecture:
        embed  → Linear(input_dim → hidden_dim) + ReLU
        layer1 → DeepSetLayer(hidden_dim)           (equivariant)
        layer2 → DeepSetLayer(hidden_dim)           (equivariant)
        rho    → max-pool over set dim (invariant)
               → Linear(hidden_dim → hidden_dim) + ReLU
               → Linear(hidden_dim → output_dim)

    Args:
        input_dim  : 4  — each ellipsoid encoded as (cx, cy, a, b)
        hidden_dim : 128
        output_dim : 256

    Forward:
        x : [B, N, 4]  (N=5 for this project; pad absent ellipsoids with zeros)
        returns [B, output_dim]
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layer1    = DeepSetLayer(hidden_dim)
        self.layer2    = DeepSetLayer(hidden_dim)
        self.rho       = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: [B, N, 4]
        x = F.relu(self.embedding(x))   # [B, N, hidden_dim]
        x = self.layer1(x)              # [B, N, hidden_dim]
        x = self.layer2(x)              # [B, N, hidden_dim]
        scene = x.max(dim=1)[0]         # [B, hidden_dim]  — invariant pooling
        return self.rho(scene)          # [B, output_dim]


# ---------------------------------------------------------------------------
# Temporal U-Net with ellipsoid conditioning
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """
    MPD-style 1D temporal U-Net for trajectory denoising, adapted for VE-SDE
    with additional DeepSet-based ellipsoid obstacle conditioning.

    Conditioning vector fed to every ResidualTemporalBlock:
        c_emb = cat(TimeEncoder(sigma), Linear(cat(x_start,x_goal)), EllipsoidFiLMEncoder(ellipsoids))
        cond_dim = time_emb_dim + conditioning_embed_dim + ellipsoid_output_dim

    Args:
        state_dim              : spatial dims per waypoint (2 for x,y)
        T_steps                : number of waypoints (trajectory horizon)
        unet_input_dim         : base channel width (32)
        dim_mults              : channel multipliers per U-Net level ((1,2,4))
        time_emb_dim           : output dim of the sigma (time) encoder (32)
        conditioning_embed_dim : dim to project cat(x_start,x_goal) into (4)
        ellipsoid_hidden_dim   : hidden dim of EllipsoidFiLMEncoder (128)
        ellipsoid_output_dim   : output dim of EllipsoidFiLMEncoder (256)
    """

    def __init__(
        self,
        state_dim:              int   = 2,
        T_steps:                int   = 64,
        unet_input_dim:         int   = 32,
        dim_mults:              tuple = (1, 2, 4),
        time_emb_dim:           int   = 32,
        conditioning_embed_dim: int   = 4,
        ellipsoid_hidden_dim:   int   = 128,
        ellipsoid_output_dim:   int   = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.T_steps   = T_steps

        # ---- Conditioning ----
        self.context_proj = nn.Linear(state_dim * 2, conditioning_embed_dim)
        self.time_mlp     = TimeEncoder(dim=32, dim_out=time_emb_dim)
        self.ellipsoid_encoder = EllipsoidFiLMEncoder(
            input_dim=4,
            hidden_dim=ellipsoid_hidden_dim,
            output_dim=ellipsoid_output_dim,
        )

        # cond_dim fed to every ResidualTemporalBlock
        cond_dim = time_emb_dim + conditioning_embed_dim + ellipsoid_output_dim

        # ---- Channel dims ----
        dims   = [state_dim, *[unet_input_dim * m for m in dim_mults]]
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
        ellipsoids: torch.Tensor,   # [B, 5, 4]    ellipsoid set (cx,cy,a,b); zeros = null
    ) -> torch.Tensor:
        """Returns predicted noise eps : [B, T, 2]."""

        t_emb        = self.time_mlp(sigma)                                    # [B, time_emb_dim]
        ctx          = self.context_proj(torch.cat([x_start, x_goal], dim=-1)) # [B, cond_emb_dim]
        ellipsoid_emb = self.ellipsoid_encoder(ellipsoids)                     # [B, ellipsoid_output_dim]
        c_emb        = torch.cat([t_emb, ctx, ellipsoid_emb], dim=-1)          # [B, cond_dim]

        x = rearrange(x, 'b t d -> b d t')   # [B, 2, T]

        skips = []
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, c_emb)
            x = resnet2(x, c_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c_emb)
        x = self.mid_block2(x, c_emb)

        for resnet1, resnet2, upsample in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet1(x, c_emb)
            x = resnet2(x, c_emb)
            x = upsample(x)

        x = self.final_conv(x)                # [B, 2, T]
        return rearrange(x, 'b d t -> b t d') # [B, T, 2]


# Alias for backward-compat
ScoreNet = TemporalUnet
