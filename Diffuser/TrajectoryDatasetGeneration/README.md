# Trajectory Dataset Generation

This directory encapsulates a generic, purely parameterized pipeline for generating 2D collision-avoiding Cartesian trajectory routes seamlessly weaving through dynamic sets of **Degree-4 Hyper-Ellipsoidal** obstacle fields utilizing informed RRT* baselines. All pathways are encoded cleanly into `Minari` `.hdf5` distributions mapping seamlessly to HDF5 architectures for PyTorch Diffuser ingestion workflows.

## Process Documentation
The generation core is powered by `trajectory_generator.py`, which directly bridges constraints dynamically derived from `torch_robotics`.
* **The Concept**: Randomly constructs a geometric map defined by `[-X, X]` configurations (e.g. `UMaze` or `LargeMaze`) scaling arbitrary bounds. The script populates independent physical barriers using Generalized $L_4$ SDF continuous definitions instead of standard collision radii.
* **The Algorithms (InfRRT* & GPMP2)**: The initial graph layout optimizes non-holonomic distributions across the field using `Informed RRT*` (which biases samples into ellipsoidal subsets natively minimizing boundary route discovery). The topology discretely found is then sequentially optimized by `GPMP2` across the smooth $L_4$ cost-space curves to resolve 0.08 physical clearance guarantees.

## Directory Workflow & Code Placement

* `trajectory_generator.py`: The single standalone core Python distribution defining `EnvUMazeRandomEllipsoids2D` and `EnvLargeMazeRandomEllipsoids2D` along with the `generate_trajectory` pipeline hook.
* `generate_dataset.ipynb`: Standalone execution mapping explicitly parameterized variables (`target_episodes`, `env_class`, `rrt_max_time`) which iteratively loops over randomized boundaries continuously writing Minari bindings per structure.
* `visualize_dataset.ipynb`: A standalone local Minari array un-packer dynamically mapping randomly sampled visual indices dynamically over Matplotlib layouts matching the native structures saved natively to cache.
* `report.pdf` / `report.tex`: Explicit mathematical details validating $L_4$ geometric boundaries and optimization metrics explicitly justifying scaling.

## How to Use

### 1. Generating Custom Datasets
Launch `generate_dataset.ipynb` and define the top-level parameters:
```python
ENV_NAME = "LargeMaze"  # Target environment bounds (UMaze or LargeMaze)
TARGET_EPISODES = 5    # The absolute sequence generation queue size
DATASET_ID = "custom/mpd_largemaze_ellipsoids-v1" 
```
Clicking "Run All" sequentially binds identical HDF5 array shapes using continuous `np.tile` paddings safely bypassing ragged shape array configurations.

**Where is the output stored?**
Minari natively intercepts local executions mapping datasets uniquely bound into `~/.minari/datasets/custom/<dataset_id>`. This allows instant environment integrations seamlessly via `minari.load_dataset(DATASET_ID)`.

### 2. Validating Trajectories
Launch `visualize_dataset.ipynb` and configure:
```python
DATASET_ID = "custom/mpd_largemaze_ellipsoids-v1"
NUM_SAMPLES = 5  # Total indices randomized out of the hdf5 bundle
```
Executing evaluates exactly randomized index episodes mapping corresponding starting topologies and bounding sequences properly into `matplotlib.pyplot` blocks locally shown.
