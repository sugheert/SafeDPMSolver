# Implementation Plan: Local Spatial Sampling for Diffusion Planning

## 1. Architectural Discrepancy & Objective
Currently, `score_net_ellipsoids.py` utilizes an `EllipsoidFiLMEncoder` to compress $N$ analytic ellipsoids into a single 256-dimensional global context vector. This vector is injected into the `TemporalUnet` via FiLM modulation.

The reference paper (*Joint Localization and Planning Using Diffusion*, Lao Beyer & Karaman, 2024) proposes a fundamentally different approach for obstacle avoidance. Instead of global modulation, it relies on **local spatial conditioning** (early fusion):
1. An environment map is processed into a dense feature grid.
2. For every timestep, the spatial coordinates of the trajectory waypoints $X^{(t)}$ are used to sample this feature grid.
3. The sampled local features are concatenated directly to the input pose encodings before passing through the denoising U-Net.

**Objective:** Refactor `score_net_ellipsoids.py` and `samplers_ellipsoids_cfg.py` to replace global FiLM ellipsoid conditioning with trajectory-aligned local feature sampling.

## 2. The "Lightweight" Local PointNet Strategy
Because a Control Barrier Function (CBF) is strictly enforcing safety during inference (using the linear Euclidean SDF to resolve gradient-vanishing issues), computing explicit analytic SDFs *inside* the U-Net during training introduces redundant computational overhead.

**The Compromise:**
We implement a PointNet-style architecture that acts as a continuous local encoder. It avoids explicitly rasterizing a grid and avoids calculating the exact SDF proxy. Instead, it computes the **raw relative geometry** ($\Delta x$, $\Delta y$, $a$, $b$) between each waypoint and each obstacle. The network learns a spatial representation from these raw inputs, providing the U-Net with enough local spatial awareness to generate a nominally safe path, which the CBF then refines.

### Mathematical Formulation
For each trajectory waypoint $x_t \in \mathbb{R}^2$ and each obstacle $o_i = (c_x, c_y, a, b)$, compute:
$$f_{t,i} = [x_t - c_x, y_t - c_y, a, b] \in \mathbb{R}^{4}$$

Process through a shared MLP ($\lambda_\theta$), pool symmetrically over the $N$ obstacles, and project ($\gamma_\phi$) to form the local conditioning vector $l_t$:
$$e_{t,i} = \lambda_\theta(f_{t,i})$$
$$z_t = \max_{i=1}^{N} (e_{t,i})$$
$$l_t = \gamma_\phi(z_t) \in \mathbb{R}^{d_{local}}$$

---

## 3. PyTorch Architecture (`LightweightLocalEncoder`)

To be added to `score_net_ellipsoids.py`, replacing `EllipsoidFiLMEncoder`.

```python
import torch
import torch.nn as nn

class LightweightLocalEncoder(nn.Module):
    """
    Computes local spatial features for a trajectory relative to a set of ellipsoids.
    Avoids explicit SDF computation, relying on raw relative geometry to save compute.
    """
    def __init__(self, d_hidden: int = 128, d_local: int = 64):
        super().__init__()
        
        # λ_theta: Shared MLP for pairwise relative features
        # Input: 4 dims (dx, dy, a, b)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )
        
        # γ_phi: Final projection after max-pooling
        self.proj_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_local)
        )

    def forward(self, x: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2] (noisy trajectory waypoints)
        obstacles: [B, N, 4] (ellipsoid params: cx, cy, a, b)
        returns: [B, T, d_local]
        """
        B, T, _ = x.shape
        _, N, _ = obstacles.shape
        
        if N == 0:
            # Unconditional / No obstacles: return zero features
            return torch.zeros(B, T, self.proj_mlp[-1].out_features, device=x.device)

        # 1. Expand tensors to compute pairwise interactions [B, T, N, ...]
        x_expanded = x.unsqueeze(2).expand(B, T, N, 2)
        obs_expanded = obstacles.unsqueeze(1).expand(B, T, N, 4)
        
        cx = obs_expanded[..., 0]
        cy = obs_expanded[..., 1]
        a = obs_expanded[..., 2]
        b = obs_expanded[..., 3]
        
        dx = x_expanded[..., 0] - cx
        dy = x_expanded[..., 1] - cy
        
        # Stack raw relative features -> [B, T, N, 4]
        rel_features = torch.stack([dx, dy, a, b], dim=-1)
        
        # 2. Apply point-wise MLP -> [B, T, N, d_hidden]
        e = self.point_mlp(rel_features)
        
        # 3. Symmetric Aggregation (Max Pool over N) -> [B, T, d_hidden]
        z = torch.max(e, dim=2)[0]
        
        # 4. Final Projection -> [B, T, d_local]
        l_t = self.proj_mlp(z)
        
        return l_t
```

---

## 4. Workflows & Refactoring Steps

### Phase 1: U-Net Refactoring (`score_net_ellipsoids.py`)

1. **Swap Encoders:**
   * Remove the `DeepSetLayer` and `EllipsoidFiLMEncoder` classes.
   * Instantiate the new local encoder in the `TemporalUnet` initialization:
     ```python
     self.local_encoder = LightweightLocalEncoder(d_hidden=128, d_local=64)
     ```

2. **Update the Global Context Dimension:**
   * Remove the ellipsoid dimension from the global conditioning vector `c_emb`.
     ```python
     # cond_dim fed to every ResidualTemporalBlock
     cond_dim = time_emb_dim + conditioning_embed_dim 
     ```

3. **Implement Early Fusion in `forward`:**
   * Compute the local features dynamically for the current noisy trajectory `x` and concatenate them before the downsampling path begins:
     ```python
     from einops import rearrange
     
     def forward(self, x, sigma, x_start, x_goal, ellipsoids):
         # 1. Global Conditioning (Time + Start/Goal)
         t_emb = self.time_mlp(sigma)
         ctx = self.context_proj(torch.cat([x_start, x_goal], dim=-1))
         c_emb = torch.cat([t_emb, ctx], dim=-1) # [B, cond_dim]

         # 2. Local Conditioning (Early Fusion)
         local_features = self.local_encoder(x, ellipsoids) # [B, T, d_local=64]
         x_fused = torch.cat([x, local_features], dim=-1)   # [B, T, 66]

         # 3. Rearrange for 1D Convolutions
         x_in = rearrange(x_fused, 'b t d -> b d t')        # [B, 66, T]
         
         # ... proceed to downsampling using x_in ...
     ```

4. **Adjust Input Convolution Channels:**
   * The very first convolution in the network (likely the first `ResidualTemporalBlock` in `self.downs` or a newly added initial projection layer) currently expects `state_dim` (2) as its input. 
   * Update this to expect `state_dim + d_local` (66) input channels.

### Phase 2: Sampler Impact (`samplers_ellipsoids_cfg.py`)

Because this early fusion approach calculates features dynamically inside the U-Net's `forward` pass based on the input `ellipsoids`, **existing CFG logic requires no architectural changes**. 

* During the unconditional pass, passing `null_ellipsoids` (zeros) will cause the `LightweightLocalEncoder` to cleanly output zero-features.
* The `dpm_solver_1_cfg_sample` and `dpm_solver_1_cbf_cfg_sample` routines will pass `eps_guided` to the solver step exactly as they currently do. 

---

## 5. Paper References
* **Obstacle Map Encoding:** Section III-B, Equation 7.
* **Local Conditioning Strategy:** Figure 2. Visually represents spatial querying, adapting naturally from a rasterized grid approach to our continuous `LightweightLocalEncoder`.
* **Feature Concatenation:** Section III-B. Features are appended to the input pose encodings $f_{in}$, rather than applied via FiLM.