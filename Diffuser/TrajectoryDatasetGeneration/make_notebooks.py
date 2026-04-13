import nbformat as nbf

# 1. GENERATE DATASET NOTEBOOK
nb_gen = nbf.v4.new_notebook()

code_gen_params = """# PARAMETERS
# Select environment: 'EnvUMazeRandomEllipsoids2D' or 'EnvLargeMazeRandomEllipsoids2D'
ENVIRONMENT_NAME = 'EnvLargeMazeRandomEllipsoids2D' 
TARGET_EPISODES = 5
DATASET_ID = "custom/mpd_largemaze_ellipsoids-v1"
RRT_MAX_TIME = 8.0 # Max time to spend finding a path per episode
MAX_OBSTACLES = 5 # Dynamic padding constraint bounds
DIST_FACTOR = 0.25 # Minimum start-to-goal diagonal margin bounds

# Where datasets are stored — set this explicitly so you always know where your data is.
# Default: ~/.minari/datasets (system store, persistent across sessions)
# Uncomment to use a local folder instead:
# DATA_DIR = "/home/earth/SDPMSP/Diffuser/TrajectoryDatasetGeneration/datasets"
DATA_DIR = None  # None = use the default system Minari store (~/.minari/datasets)
"""

code_gen_setup = """import os
import sys
import torch
import numpy as np
from gymnasium.spaces import Box
import minari
from minari import EpisodeData

# Mapping backend paths natively
sys.path.append('/home/earth/MPD/mpd-public')
sys.path.append('/home/earth/MPD/mpd-public/deps/torch_robotics')

from trajectory_generator import EnvUMazeRandomEllipsoids2D, EnvLargeMazeRandomEllipsoids2D, generate_trajectory
from torch_robotics.robots import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask

tensor_args = {'device': 'cpu', 'dtype': torch.float32}

# Configure data storage path
if DATA_DIR is not None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.environ['MINARI_DATASETS_PATH'] = os.path.abspath(DATA_DIR)
    print(f"Storing datasets in: {os.path.abspath(DATA_DIR)}")
else:
    print(f"Storing datasets in system Minari store (~/.minari/datasets)")
"""

code_gen_loop = """# Dataset Generation Loop
print(f"===========================================")
print(f"Beginning dataset generation for: {DATASET_ID}")

env_class = EnvLargeMazeRandomEllipsoids2D if ENVIRONMENT_NAME == 'EnvLargeMazeRandomEllipsoids2D' else EnvUMazeRandomEllipsoids2D

ep = 0
episodes_buffer = []

while ep < TARGET_EPISODES:
    print(f"--- Episode {ep+1} ---")
    env = env_class(tensor_args=tensor_args, max_obstacles=MAX_OBSTACLES)
    robot = RobotPointMass(q_limits=env.limits, tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, obstacle_cutoff_margin=0.08, tensor_args=tensor_args)

    min_dist = torch.norm(env.limits[1] - env.limits[0]) * DIST_FACTOR 
    
    found_start_goal = False
    for _ in range(500):
        q_free = task.random_coll_free_q(n_samples=2)
        start_p, goal_p = q_free[0], q_free[1]
        if torch.norm(start_p - goal_p) > min_dist:
            start_pos, goal_pos = start_p, goal_p
            found_start_goal = True
            break
    
    if not found_start_goal:
        continue
        
    success, free_trajs = generate_trajectory(env, robot, task, start_pos, goal_pos, tensor_args, gpmp_opt_iters=200, rrt_max_time=RRT_MAX_TIME)
    
    if success:
        print(f"  - Generating EpisodeData ({ep+1}/{TARGET_EPISODES})")
        traj = free_trajs[0].detach().cpu().numpy()
        actions = traj[:, 2:4][:-1] if traj.shape[-1] == 4 else traj[1:] - traj[:-1]
        observations = traj
        rewards = np.zeros(len(actions))
        terminations = np.array([False]*(len(actions)-1) + [True])
        truncations = np.array([False]*len(actions))

        padded_centers = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)
        padded_centers[:len(env.ellipsoids_centers)] = env.ellipsoids_centers
        padded_radii = np.zeros((MAX_OBSTACLES, 2), dtype=np.float32)
        padded_radii[:len(env.ellipsoids_radii)] = env.ellipsoids_radii

        centers_tile = np.tile(padded_centers, (len(actions), 1, 1))
        radii_tile = np.tile(padded_radii, (len(actions), 1, 1))
        start_tile = np.tile(start_pos.cpu().numpy().astype(np.float32), (len(actions), 1))
        goal_tile = np.tile(goal_pos.cpu().numpy().astype(np.float32), (len(actions), 1))

        episode_info = {
            "ellipsoids_centers": centers_tile,
            "ellipsoids_radii": radii_tile,
            "start_pos": start_tile,
            "goal_pos": goal_tile,
        }

        episode_data = EpisodeData(
            id=ep,
            observations=observations.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            terminations=terminations,
            truncations=truncations,
            infos=episode_info
        )
        object.__setattr__(episode_data, 'seed', None)
        object.__setattr__(episode_data, 'options', None)
        episodes_buffer.append(episode_data)
        ep += 1

print(f"Saving compiled Minari Dataset: {DATASET_ID} to ./data")
obs_space = Box(low=-np.inf, high=np.inf, shape=(observations.shape[-1],), dtype=np.float32)
act_space = Box(low=-np.inf, high=np.inf, shape=(actions.shape[-1],), dtype=np.float32)

# Append to existing dataset, or create a new one if it doesn't exist
# Robust existence check — list_local_datasets crashes if any entry has a malformed ID
def dataset_exists(dataset_id):
    try:
        return dataset_id in minari.list_local_datasets()
    except (TypeError, ValueError):
        # Fallback: check the HDF5 file directly
        import pathlib
        data_path = pathlib.Path(os.environ['MINARI_DATASETS_PATH']) / dataset_id / 'data' / 'main_data.hdf5'
        return data_path.exists()

if dataset_exists(DATASET_ID):
    print(f"  Dataset already exists — appending {TARGET_EPISODES} new episodes...")
    existing = minari.load_dataset(DATASET_ID)
    # total_episodes can be None if metadata wasn't fully flushed — count from storage directly
    episodes_before = existing.total_episodes
    if episodes_before is None:
        import h5py
        with h5py.File(existing._data._file_path, 'r') as f:
            episodes_before = len(f.keys())
    episodes_before = int(episodes_before)
    # Re-ID episodes sequentially from the existing count so they don't overwrite
    for i, ep_data in enumerate(episodes_buffer):
        object.__setattr__(ep_data, 'id', episodes_before + i)
    existing._data.update_episodes(episodes_buffer)
    dataset = minari.load_dataset(DATASET_ID)
    print(f"  Episodes before: {episodes_before} → after: {dataset.total_episodes}")
else:
    print(f"  Creating new dataset...")
    dataset = minari.create_dataset_from_buffers(
        dataset_id=DATASET_ID,
        buffer=episodes_buffer,
        observation_space=obs_space,
        action_space=act_space,
        algorithm_name="InfRRTStar_GPMP2",
        author="MPD Parametric Pipeline"
    )
    print(f"  New dataset created with {dataset.total_episodes} episodes.")
"""

nb_gen.cells = [
    nbf.v4.new_markdown_cell("# Trajectory Dataset Generator\nExecuting paths explicitly via dynamic `InfRRTStar` models bounded across generic arrays natively written into standalone Minari `./data/` directories."),
    nbf.v4.new_code_cell(code_gen_params),
    nbf.v4.new_code_cell(code_gen_setup),
    nbf.v4.new_code_cell(code_gen_loop)
]
with open('generate_dataset.ipynb', 'w') as f:
    nbf.write(nb_gen, f)


# 2. VISUALIZE DATASET NOTEBOOK
nb_viz = nbf.v4.new_notebook()

code_viz_params = """# PARAMETERS
DATASET_ID = "custom/mpd_largemaze_ellipsoids-v1"
NUM_SAMPLES = 5 # Episodes to visualize randomly
ENVIRONMENT_NAME = 'EnvLargeMazeRandomEllipsoids2D' 
"""

code_viz_setup = """import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import minari

sys.path.append('/home/earth/MPD/mpd-public')
sys.path.append('/home/earth/MPD/mpd-public/deps/torch_robotics')

from trajectory_generator import EnvUMazeRandomEllipsoids2D, EnvLargeMazeRandomEllipsoids2D
tensor_args = {'device': 'cpu', 'dtype': torch.float32}

# Load the mapped dataset locally:
local_data = os.path.abspath('./data')
os.environ['MINARI_DATASETS_PATH'] = local_data
dataset = minari.load_dataset(DATASET_ID)
print(f"Dataset successfully loaded possessing {dataset.total_episodes} random trajectories.")
"""

code_viz_loop = """# Fetch completely randomized episodes
np.random.seed(None)
sampled_ids = np.random.choice(dataset.episode_indices, size=NUM_SAMPLES, replace=False)

env_class = EnvLargeMazeRandomEllipsoids2D if ENVIRONMENT_NAME == 'EnvLargeMazeRandomEllipsoids2D' else EnvUMazeRandomEllipsoids2D

for ep_id in sampled_ids:
    episode = dataset[ep_id]
    
    # Environment info constants are tiled across time steps T.
    # Take the exact snapshot from the very final mapping natively bounds T=0.
    start_pos = np.array(episode.infos['start_pos'][0])
    goal_pos = np.array(episode.infos['goal_pos'][0])
    
    centers = np.array(episode.infos['ellipsoids_centers'][0])
    radii = np.array(episode.infos['ellipsoids_radii'][0])
    
    # Safely truncate the max_obstacle sequence padded zeroes out
    valid_mask = np.sum(np.abs(radii), axis=-1) > 1e-4
    centers = centers[valid_mask]
    radii = radii[valid_mask]

    # Dynamically boot an env layout container strictly for visualization matrices
    env = env_class(tensor_args=tensor_args)
    env.ellipsoids_centers = centers
    env.ellipsoids_radii = radii
    # Map the custom centers locally bounding `MultiEllipsoidField`
    env.obj_fixed_list[0].fields[-1].centers = torch.tensor(centers, **tensor_args)
    env.obj_fixed_list[0].fields[-1].radii = torch.tensor(radii, **tensor_args)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    env.render(ax)
    
    # Plot Trajectory sequence (Observations are N x 4 or N x 2)
    obs = episode.observations
    ax.plot(obs[:, 0], obs[:, 1], color='blue', linewidth=2, label='Optimized GPMP2 Trajectory')
    
    ax.scatter(start_pos[0], start_pos[1], color='green', s=100, zorder=5, label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], color='red', s=100, marker='*', zorder=5, label='Goal')
    
    ax.set_title(f"Episode {ep_id}")
    ax.legend(loc='upper right')
    plt.show()
"""

nb_viz.cells = [
    nbf.v4.new_markdown_cell("# Visualize Trajectory Datasets\nAutomatically parses randomized Minari distributions extracting specific `EpisodeData` mapping `InfRRTStar` generated routes safely integrating geometric representations globally."),
    nbf.v4.new_code_cell(code_viz_params),
    nbf.v4.new_code_cell(code_viz_setup),
    nbf.v4.new_code_cell(code_viz_loop)
]
with open('visualize_dataset.ipynb', 'w') as f:
    nbf.write(nb_viz, f)

print("Generated generate_dataset.ipynb and visualize_dataset.ipynb successfully!")
