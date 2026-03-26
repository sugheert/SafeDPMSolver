# PLAN

## Task 1: Mathematical Refactoring (`CBF/trajectory_cbf.py`)
1. Implement `softplus_distance_circle` computing `d~_{ij}` from `d_{ij}` using parameter `c`.
2. Update `waypoint_cbf` to incorporate the numerical stabilization factor `d~_{min,i}`.
3. Update `trajectory_cbf` to use numerically stabilized weights `sigma_i` using `h_{min}`.
4. Implement the new gradient functions `grad_hwi_dwi` and `grad_hXt_dwi` utilizing the sigmoid gate derivative and the normalized weights `alpha_{ij}` and `sigma_i`.

## Task 2: Codebase Modularization
1. Parse the cells in `notebooks/train_and_sample_circles_copy.ipynb` to identify code duplications.
2. Abstract the `EulerMaruyamaSamplerCBF` into modular Python file `models/samplers.py` for direct import.
3. Create `utils/data.py` and `utils/visualization.py` storing the dataset loaders and plotting helpers respectively.
4. Refactor `train_and_sample_circles_copy.ipynb` to import and orchestrate these isolated components cleanly, removing redundant cell clutter.

## Task 3: Visualizer API Backend (`app.py`)
1. Setup a FastAPI (or similar lightweight framework) app in a `visualizer/` folder.
2. Add a `/api/models` endpoint checking the `checkpoints/` directory to return available `.pt` filenames.
3. Add a `/api/optimise` endpoint accepting the CBF parameters (`c`, `k1`, `k2`, `r`, `gamma_delta`) and selected model. It invokes the score network, generates a shared initial prior, and performs both the uncontrolled (Plain DPM-Solver-1) and controlled (Safe DPM-Solver-1) ancestral loop for `N` steps simultaneously. Return the step-by-step history of both trajectories alongside the safety math payloads to the frontend.

## Task 4: Interactive Frontend (`static/`)
1. Create `index.html` mocking the strict UI layout from `simulationRequirements.md` (Toolbar, Trajectory Canvas, Inspector Table, Parameter controls including `gamma_delta`).
2. Create `style.css` defining the square interactive canvas, sticky header table, input configurations, etc.
3. Create `app.js` configuring D3.js or basic Canvas to render both the Uncontrolled (dashed) and Safe (solid) trajectory waypoints (safe/violated styling, gradient arrows based on Safe path).
4. Implement interactive mouse event listeners handling standard node dragging and Gaussian Spline deformations (Ctrl+Drag) for the Safe trajectory.
5. Hook up the Run button to ping the `/api/optimise` backend, caching the step history of BOTH trajectories to be iterated interactively via the UI Step Scrubber.
