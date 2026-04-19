"""
score_net_ellipsoids_rasterizedlocalfeats.py — TemporalUnet with local feature sampling.

Replaces the global average-pool FiLM conditioning of score_net_ellipsoids.py with
per-waypoint local feature sampling:

    1. MapEncoder encodes the binary occupancy map into a spatial feature map
       [B, local_dim, H, W] (full resolution via Micro U-Net).
    2. sample_local_features bilinearly samples that map at each noisy waypoint
       position → [B, T, local_dim].
    3. The local features are concatenated to the waypoint input before the U-Net:
       x_in [B, T, state_dim + local_dim] → rearranged to [B, state_dim+local_dim, T].
    4. The global FiLM vector carries only sigma + start/goal context (no map term).

Architecture:
    occ_map [B, 1, H, W]
        → MapEncoder            → feat_map [B, local_dim, H, W]
        → sample_local_features ← noisy waypoints [B, T, 2]
        → local_feat [B, T, local_dim]
        → cat([x, local_feat])  → x_in [B, T, state_dim+local_dim]
        → rearrange             → [B, state_dim+local_dim, T]
        → Temporal U-Net (FiLM from sigma + start/goal only)

Modified TemporalUnet.forward signature:
    forward(x, sigma, x_start, x_goal, occ_map, cached_feat_map=None)

Pass cached_feat_map to skip re-encoding during inference (map is static per episode).
Null (unconditional) input for CFG: pass an all-zeros bitmap for occ_map.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_norm_n_groups(n_channels, target_n_groups=8):
    if n_channels < target_n_groups:
        return 1
    for n_groups in range(target_n_groups, target_n_groups + 10):
        if n_channels % n_groups == 0:
            return n_groups
    return 1


# ---------------------------------------------------------------------------
# Time / sigma encoding
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
# Conv building blocks
# ---------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
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
        h = self.blocks[0](x) + self.cond_mlp(c)
        h = self.blocks[1](h)
        return h + self.residual_conv(x)


# ---------------------------------------------------------------------------
# DeepSet ellipsoid encoder (retained for reference, not used by TemporalUnet)
# ---------------------------------------------------------------------------

class DeepSetLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lam = nn.Linear(dim, dim)
        self.gam = nn.Linear(dim, dim)

    def forward(self, x):
        local_part  = self.lam(x)
        global_max  = x.max(dim=1, keepdim=True)[0]
        global_part = self.gam(global_max)
        return F.relu(local_part + global_part)


class EllipsoidFiLMEncoder(nn.Module):
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
        x = F.relu(self.embedding(x))
        x = self.layer1(x)
        x = self.layer2(x)
        scene = x.max(dim=1)[0]
        return self.rho(scene)


# ---------------------------------------------------------------------------
# CNN occupancy-map encoder
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MapEncoder(nn.Module):
    """
    Encodes a binary occupancy bitmap into a full-resolution spatial feature map.

    Input  : [B, 1, H, W]
    Output : [B, local_dim, H, W]   (64×64 for H=W=64)

    Micro U-Net with skip connections preserves spatial resolution end-to-end.
    """
    def __init__(self, local_dim: int = 16):
        super().__init__()
        self.down1   = DoubleConv(1, 16)
        self.pool1   = nn.MaxPool2d(2)
        self.down2   = DoubleConv(16, 32)
        self.pool2   = nn.MaxPool2d(2)
        self.bottle  = DoubleConv(32, 64)
        self.up1     = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_u1 = DoubleConv(64, 32)   # 32 up + 32 skip
        self.up2     = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_u2 = DoubleConv(32, 16)   # 16 up + 16 skip
        self.out     = nn.Conv2d(16, local_dim, kernel_size=1)

    def forward(self, occ_map: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(occ_map)              # [B, 16, H,   W  ]
        s2 = self.down2(self.pool1(s1))       # [B, 32, H/2, W/2]
        x  = self.bottle(self.pool2(s2))      # [B, 64, H/4, W/4]
        x  = self.conv_u1(torch.cat([self.up1(x), s2], dim=1))  # [B, 32, H/2, W/2]
        x  = self.conv_u2(torch.cat([self.up2(x), s1], dim=1))  # [B, 16, H,   W  ]
        return self.out(x)                    # [B, local_dim, H, W]


class GlobalPoolHead(nn.Module):
    """Retained for reference; not used by TemporalUnet in this file."""
    def __init__(self, local_dim: int = 16, global_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(local_dim, global_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(feat_map))


# ---------------------------------------------------------------------------
# Local feature sampler
# ---------------------------------------------------------------------------

def sample_local_features(
    feat_map:  torch.Tensor,   # [B, C, H', W']
    positions: torch.Tensor,   # [B, T, 2]  in [-1, 1]
) -> torch.Tensor:             # [B, T, C]
    """
    Bilinearly samples the spatial feature map at each waypoint position.

    F.grid_sample expects a grid of shape [B, H_out, W_out, 2].
    We treat T waypoints as W_out and set H_out=1.
    """
    clamped_pos = torch.clamp(positions, -1.1, 1.1)
    grid = clamped_pos.unsqueeze(1) 
    feats = F.grid_sample(
        feat_map, grid,
        mode='bilinear',
        align_corners=True,
        padding_mode='border',
    )                                                     # [B, C, 1, T]
    return feats.squeeze(2).permute(0, 2, 1)              # [B, T, C]


# ---------------------------------------------------------------------------
# Temporal U-Net with local feature sampling
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """
    MPD-style 1D temporal U-Net for trajectory denoising with local feature sampling.

    Map conditioning:
        - MapEncoder produces feat_map [B, local_dim, H/8, W/8].
        - sample_local_features samples feat_map at each noisy waypoint → [B, T, local_dim].
        - Local features are concatenated to waypoint input; U-Net sees [B, state_dim+local_dim, T].
        - Global FiLM carries only sigma + start/goal (no map term).

    Args:
        state_dim              : spatial dims per waypoint (2 for x,y)
        T_steps                : number of waypoints (trajectory horizon)
        unet_input_dim         : base channel width (32)
        dim_mults              : channel multipliers per U-Net level ((1,2,4))
        time_emb_dim           : output dim of the sigma (time) encoder (32)
        conditioning_embed_dim : dim to project cat(x_start,x_goal) into (4)
        local_dim              : MapEncoder feature channels, also appended to waypoints (16)
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
    ):
        super().__init__()
        self.state_dim = state_dim
        self.T_steps   = T_steps
        self.local_dim = local_dim

        # ---- Conditioning (sigma + start/goal only; map handled locally) ----
        self.context_proj = nn.Linear(state_dim * 2, conditioning_embed_dim)
        self.time_mlp     = TimeEncoder(dim=32, dim_out=time_emb_dim)
        self.map_encoder  = MapEncoder(local_dim=local_dim)

        cond_dim = time_emb_dim + conditioning_embed_dim   # 32 + 4 = 36

        # ---- Channel dims — first dim expands by local_dim ----
        dims   = [state_dim + local_dim, *[unet_input_dim * m for m in dim_mults]]
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
        x:               torch.Tensor,        # [B, T, 2]      noisy trajectory
        sigma:           torch.Tensor,        # [B]             noise level
        x_start:         torch.Tensor,        # [B, 2]          start condition
        x_goal:          torch.Tensor,        # [B, 2]          goal condition
        occ_map:         torch.Tensor = None, # [B, 1, H, W]    occupancy map; ignored if cached
        cached_feat_map: torch.Tensor = None, # [B, C, H', W']  pre-computed feat map for inference
    ) -> torch.Tensor:
        """Returns predicted noise eps : [B, T, 2]."""

        # 1. Global conditioning: sigma + start/goal only
        t_emb = self.time_mlp(sigma)
        ctx   = self.context_proj(torch.cat([x_start, x_goal], dim=-1))
        c_emb = torch.cat([t_emb, ctx], dim=-1)   # [B, 36]

        # 2. Spatial feature map (encode once; reuse cached during inference)
        feat_map = cached_feat_map if cached_feat_map is not None else self.map_encoder(occ_map)

        # 3. Sample feat_map at each waypoint position
        positions  = x[..., :2]                                    # [B, T, 2]
        local_feat = sample_local_features(feat_map, positions)    # [B, T, local_dim]

        # 4. Concatenate local features and rearrange for Conv1d
        x_in = torch.cat([x, local_feat], dim=-1)                  # [B, T, state_dim+local_dim]
        x_in = rearrange(x_in, 'b t d -> b d t')                   # [B, state_dim+local_dim, T]

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

        x_in = self.final_conv(x_in)                # [B, 2, T]
        return rearrange(x_in, 'b d t -> b t d')    # [B, T, 2]


ScoreNet = TemporalUnet
