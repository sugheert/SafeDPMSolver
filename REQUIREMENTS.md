# REQUIREMENTS

## 1. Mathematical Update: Stabilized Softplus CBF
The primary module `CBF/trajectory_cbf.py` must be refactored to implement the updated derivations:
* **Distance Transformation**: Introduce the `softplus` formulation with sharpness parameter `c` to transform the signed distance `d_{ij}` into `d~_{ij}`:
  `d~_{ij} = -c * log(1 + exp(-d_{ij}/c)) + log(2)`
* **Numerical Stabilization**: The exponential terms inside the softmin functions overflow/underflow easily. Factor out the minimum terms `h_{min}` and `d~_{min,i}` to compute strictly stabilized weights `sigma_i` (waypoint importance) and `alpha_{ij}` (obstacle importance).
* **Gradient Gating**: The gradient `\nabla_{w_i} h(w_i)` must utilize the soft-gate derivative of the softplus function: the sigmoid `\sigma(-d_{ij}/c)`. This accurately kills gradients coming from far, safe obstacles while preserving full gradients for overlapping ones.

## 2. Code Modularity
The codebase should be refactored such that logic inside notebooks acts exclusively as an execution/presentation harness.
* Extract custom samplers (e.g. Euler-Maruyama with CBF) to an isolated file `models/samplers.py`.
* Move plotting routines to a distinct `utils/visualization.py`.
* Ensure data loading and environment setup functions are importable explicitly into any scripts or API endpoints.

## 3. Interactive Web Visualizer
Build a proper web application capable of running locally on `localhost`.
* **Backend (e.g., FastAPI)**: Provide endpoints to list available models from the `checkpoints/` folder. Provide a major `/run_optimisation` endpoint that instantiates a score model, runs the diffusion/ancestral sampler loop using live CBF parameters (`c`, `k1`, `k2`, `r`), and returns every evaluated step cache (waypoint locations, step distances `d_i`, `d~_i`, waypoint-CBF `h(w_i)`, scalar weights `sigma_i`, and gradient vectors `\partial x`, `\partial y`).
* **Frontend**: Strictly follow the visual layout provided in `simulationRequirements.md`. Implement the Trajectory Canvas (~55% width) side-by-side with the Step Inspector table (~45% width). All components (Dropdowns, run buttons, scrubbers, configuration parameter sliders) must wire directly to the Backend and conditionally react to the cached data from the optimization run.

These actions will streamline agent-based optimizations and yield minimal structural conflicts during implementation.
