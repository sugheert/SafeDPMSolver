"""
score_net_ellipsoids.py — TemporalUnet conditioned on a rasterized occupancy map.

Replaces the DeepSet-based EllipsoidFiLMEncoder with a CNN MapEncoder that
reads a binary occupancy bitmap [B, 1, H, W] containing both maze walls and
ellipsoid obstacles.  The encoder produces a global scene vector [B, 256]
via average-pooling, which is used as a FiLM conditioning signal appended to
the existing sigma + start/goal conditioning.

Null (unconditional) input for CFG: pass an all-zeros bitmap.

Architecture:
    occ_map [B, 1, H, W]
        → MapEncoder  → feat_map [B, local_dim, H/8, W/8]
        → GlobalPoolHead → map_global [B, global_dim]
        → cat([time_emb, ctx, map_global]) → c_emb [B, cond_dim]
        → FiLM inject into every ResidualTemporalBlock

Modified TemporalUnet.forward signature:
    forward(x, sigma, x_start, x_goal, occ_map)
    where occ_map: [B, 1, H, W]  — binary occupancy, zeros = unconditional

Legacy EllipsoidFiLMEncoder is retained below for reference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
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
# CNN occupancy-map encoder
# ---------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """
    Encodes a binary occupancy bitmap into a spatial feature map.

    Uses the first two layer groups of a ResNet-18 backbone (conv1 through
    layer2) adapted for single-channel input, followed by a 1×1 projection.

    Input  : [B, 1, H, W]                binary occupancy map
    Output : [B, local_dim, H/8, W/8]    spatial feature map

    For H=W=64, the output spatial size is 8×8.
    The feature map is computed once per scene and reused across all
    denoising iterations.
    """
    def __init__(self, local_dim: int = 16):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        # Patch conv1 to accept single-channel (grayscale) input
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # [B, 64,  H/4,  W/4]
            resnet.layer2,   # [B, 128, H/8,  W/8]
        )
        self.proj = nn.Conv2d(128, local_dim, kernel_size=1)

    def forward(self, occ_map: torch.Tensor) -> torch.Tensor:
        # occ_map: [B, 1, H, W]
        return self.proj(self.backbone(occ_map))   # [B, local_dim, H/8, W/8]


class GlobalPoolHead(nn.Module):
    """
    Pools a spatial feature map into a single global conditioning vector.

    Input  : [B, local_dim, H', W']
    Output : [B, global_dim]
    """
    def __init__(self, local_dim: int = 16, global_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(local_dim, global_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(feat_map))   # [B, global_dim]


# ---------------------------------------------------------------------------
# Temporal U-Net with rasterized-map conditioning
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """
    MPD-style 1D temporal U-Net for trajectory denoising, conditioned on a
    rasterized occupancy map via a CNN-based global FiLM vector.

    Conditioning vector fed to every ResidualTemporalBlock:
        c_emb = cat(TimeEncoder(sigma), Linear(cat(x_start,x_goal)), GlobalPoolHead(MapEncoder(occ_map)))
        cond_dim = time_emb_dim + conditioning_embed_dim + global_dim

    Args:
        state_dim              : spatial dims per waypoint (2 for x,y)
        T_steps                : number of waypoints (trajectory horizon)
        unet_input_dim         : base channel width (32)
        dim_mults              : channel multipliers per U-Net level ((1,2,4))
        time_emb_dim           : output dim of the sigma (time) encoder (32)
        conditioning_embed_dim : dim to project cat(x_start,x_goal) into (4)
        local_dim              : MapEncoder intermediate feature channels (16)
        global_dim             : GlobalPoolHead output dim, feeds FiLM (256)
    """

    def __init__(
        self,
        state_dim:              int   = 2,
        T_steps:                int   = 64,
        unet_input_dim:         int   = 32,
        dim_mults:              tuple = (1, 2, 4),
        time_emb_dim:           int   = 32,
        conditioning_embed_dim: int   = 4,
        local_dim:              int   = 16,
        global_dim:             int   = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.T_steps   = T_steps

        # ---- Conditioning ----
        self.context_proj = nn.Linear(state_dim * 2, conditioning_embed_dim)
        self.time_mlp     = TimeEncoder(dim=32, dim_out=time_emb_dim)
        self.map_encoder  = MapEncoder(local_dim=local_dim)
        self.global_head  = GlobalPoolHead(local_dim=local_dim, global_dim=global_dim)

        # cond_dim fed to every ResidualTemporalBlock
        cond_dim = time_emb_dim + conditioning_embed_dim + global_dim

        # ---- Channel dims (U-Net input channels unchanged — no local concat) ----
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
        x:       torch.Tensor,   # [B, T, 2]     noisy trajectory
        sigma:   torch.Tensor,   # [B]            noise level
        x_start: torch.Tensor,   # [B, 2]         start condition
        x_goal:  torch.Tensor,   # [B, 2]         goal  condition
        occ_map: torch.Tensor,   # [B, 1, H, W]   binary occupancy map; zeros = unconditional
    ) -> torch.Tensor:
        """Returns predicted noise eps : [B, T, 2]."""

        t_emb      = self.time_mlp(sigma)                                     # [B, time_emb_dim]
        ctx        = self.context_proj(torch.cat([x_start, x_goal], dim=-1))  # [B, cond_emb_dim]
        feat_map   = self.map_encoder(occ_map)                                # [B, local_dim, H', W']
        map_global = self.global_head(feat_map)                               # [B, global_dim]
        c_emb      = torch.cat([t_emb, ctx, map_global], dim=-1)              # [B, cond_dim]

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
