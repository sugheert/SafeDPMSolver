import sys
import os
import torch
import numpy as np

# Add MPD paths to sys.path
sys.path.append('/home/earth/MPD/mpd-public')
sys.path.append('/home/earth/MPD/mpd-public/deps/torch_robotics')

from torch_robotics.environments.primitives import PrimitiveShapeField, ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from math import ceil

class MultiEllipsoidField(PrimitiveShapeField):
    def __init__(self, centers, radii, tensor_args=None):
        """
        Axis-aligned hyper-ellipsoids.
        Parameters
        ----------
            centers : numpy array or torch tensor (N, dim)
                Centers of the ellipsoids.
            radii : numpy array or torch tensor (N, dim)
                Radii (semi-axes) of the ellipsoids.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.radii = to_torch(radii, **self.tensor_args)

    def compute_signed_distance_impl(self, x):
        # We approximate SDF using a p-norm with p=4 for degree 4 hyper-ellipsoids
        coords_normalized = (x.unsqueeze(-2) - self.centers.unsqueeze(0)) / self.radii.unsqueeze(0)
        dists = torch.norm(coords_normalized, p=4, dim=-1) - 1.0
        # multiply by min radii to scale the SDF correctly in metric space roughly
        min_radii = torch.min(self.radii, dim=-1)[0]
        sdfs = dists * min_radii.unsqueeze(0)
        return torch.min(sdfs, dim=-1)[0]
    
    def zero_grad(self):
        self.centers.grad = None
        self.radii.grad = None

    def _is_inside(self, p, center, radii):
        return torch.sum(((p - center) / radii) ** 4) <= 1.0

    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to occupancy map
        for center, radii in zip(self.centers, self.radii):
            n_dim = len(center)
            # Find bounds
            max_r = torch.max(radii).item()
            c_r = ceil(max_r / obst_map.cell_size)
            
            c_x = ceil(center[0].item() / obst_map.cell_size)
            c_y = ceil(center[1].item() / obst_map.cell_size)

            obst_map_origin_xi = obst_map.origin[0]
            obst_map_origin_yi = obst_map.origin[1]

            c_x_cell = c_x + obst_map_origin_xi
            c_y_cell = c_y + obst_map_origin_yi

            obst_map_x_dim = obst_map.dims[0]
            obst_map_y_dim = obst_map.dims[1]

            for i in range(c_x_cell - 2 * c_r, c_x_cell + 2 * c_r):
                if i < 0 or i >= obst_map_x_dim:
                    continue
                for j in range(c_y_cell - 2 * c_r, c_y_cell + 2 * c_r):
                    if j < 0 or j >= obst_map_y_dim:
                        continue

                    p = torch.tensor([(i - obst_map_origin_xi) * obst_map.cell_size,
                                      (j - obst_map_origin_yi) * obst_map.cell_size],
                                     **self.tensor_args)

                    if self._is_inside(p, center, radii):
                        obst_map.map[j, i] += 1
                        
        return obst_map

    def render(self, ax, pos=None, ori=None, color='gray', cmap='gray', **kwargs):
        for center, radii in zip(self.centers, self.radii):
            c_np = to_numpy(center)
            r_np = to_numpy(radii)
            if pos is not None:
                c_np = c_np + to_numpy(pos)[:len(c_np)]
                
            if ax.name == '3d':
                # Skip 3D rendering for now as D4RL Maze is 2D
                pass
            else:
                # parameterize the degree 4 superellipse
                t = np.linspace(0, 2 * np.pi, 100)
                # x = c_x + r_x * sign(cos t) * |cos t|^(2/4)
                x = c_np[0] + r_np[0] * np.sign(np.cos(t)) * np.sqrt(np.abs(np.cos(t)))
                y = c_np[1] + r_np[1] * np.sign(np.sin(t)) * np.sqrt(np.abs(np.sin(t)))
                polygon = plt.Polygon(np.column_stack([x, y]), color=color, alpha=1)
                ax.add_patch(polygon)

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.robots import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask
from mp_baselines.planners.rrt_star import InfRRTStar
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner

class EnvRandomEllipsoids2D(EnvBase):
    def __init__(self,
                 name='EnvRandomEllipsoids2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 max_obstacles=4,
                 env_limit=1.0, # e.g. -1 to 1
                 seed=None,
                 **kwargs
                 ):
        if seed is not None:
             np.random.seed(seed)
             torch.manual_seed(seed)

        num_ellipsoids = np.random.randint(1, max_obstacles + 1)
        
        # We need random centers in roughly (-0.8, 0.8) so they fit inside the map
        center_limit = env_limit * 0.8
        centers = np.random.uniform(-center_limit, center_limit, size=(num_ellipsoids, 2))
        
        # Radii between 10% and 30% of env_limit so they are not too big or too small
        radii = np.random.uniform(0.1*env_limit, 0.3*env_limit, size=(num_ellipsoids, 2))
        
        self.ellipsoids_centers = centers
        self.ellipsoids_radii = radii

        obj_list = [
            MultiEllipsoidField(centers, radii, tensor_args=tensor_args)
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-env_limit, -env_limit], [env_limit, env_limit]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'random_ellipsoids')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.03,
            n_pre_samples=50000,
            max_time=15
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.02,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

class EnvUMazeRandomEllipsoids2D(EnvBase):
    def __init__(self,
                 name='EnvUMazeRandomEllipsoids2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.01,
                 max_obstacles=2, # maybe fewer in umze
                 seed=None,
                 **kwargs
                 ):
        if seed is not None:
             np.random.seed(seed)
             torch.manual_seed(seed)
             
        # UMaze mapping
        maze_map = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        
        box_centers = []
        box_sizes = []
        free_space_centers = []
        for i in range(5):
            for j in range(5):
                cx = -2.0 + j * 1.0 # x is column
                cy =  2.0 - i * 1.0 # y is row
                if maze_map[i][j] == 1:
                    box_centers.append([cx, cy])
                    box_sizes.append([1.0, 1.0])
                else:
                    free_space_centers.append([cx, cy])
                    
        num_ellipsoids = np.random.randint(1, max_obstacles + 1)
        
        # Pick random free space cells for ellipsoid centers
        chosen_indices = np.random.choice(len(free_space_centers), size=num_ellipsoids, replace=False)
        centers = []
        for idx in chosen_indices:
            # add small noise to center
            cx, cy = free_space_centers[idx]
            centers.append([cx + np.random.uniform(-0.2, 0.2), cy + np.random.uniform(-0.2, 0.2)])
        centers = np.array(centers)
        
        # Radii between 0.3 and 0.6 so they fit inside free space comfortably but are visibly substantial
        radii = np.random.uniform(0.3, 0.6, size=(num_ellipsoids, 2))
        
        self.ellipsoids_centers = centers
        self.ellipsoids_radii = radii

        obj_list = [
            MultiBoxField(np.array(box_centers), np.array(box_sizes), tensor_args=tensor_args),
            MultiEllipsoidField(centers, radii, tensor_args=tensor_args)
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-2.5, -2.5], [2.5, 2.5]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'umaze_random_ellipsoids')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.05,
            n_radius=0.1,
            n_pre_samples=50000,
            max_time=15
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.08,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=2e-5, # slightly higher repulstion
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

class EnvLargeMazeRandomEllipsoids2D(EnvBase):
    def __init__(self,
                 name='EnvLargeMazeRandomEllipsoids2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.02, # Slightly larger cell to reduce memory if needed, or stick to 0.01
                 max_obstacles=6, # More obstacles possible due to larger freespace
                 seed=None,
                 **kwargs
                 ):
        if seed is not None:
             np.random.seed(seed)
             torch.manual_seed(seed)
             
        # Large Maze mapping from D4RL
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        
        box_centers = []
        box_sizes = []
        free_space_centers = []
        rows = len(maze_map)
        cols = len(maze_map[0])
        for i in range(rows):
            for j in range(cols):
                cx = -5.5 + j * 1.0 # x is column (-6 to 6 => centers at -5.5, -4.5...)
                cy =  4.0 - i * 1.0 # y is row   (4.5 to -4.5 => centers at 4.0, 3.0...)
                if maze_map[i][j] == 1:
                    box_centers.append([cx, cy])
                    box_sizes.append([1.0, 1.0])
                else:
                    free_space_centers.append([cx, cy])
                    
        num_ellipsoids = np.random.randint(1, max_obstacles + 1)
        
        # Pick random free space cells for ellipsoid centers
        chosen_indices = np.random.choice(len(free_space_centers), size=num_ellipsoids, replace=False)
        centers = []
        for idx in chosen_indices:
            cx, cy = free_space_centers[idx]
            centers.append([cx + np.random.uniform(-0.2, 0.2), cy + np.random.uniform(-0.2, 0.2)])
        centers = np.array(centers)
        
        # Radii between 0.4 and 1.0 (large obstacles to block major passages)
        radii = np.random.uniform(0.4, 1.0, size=(num_ellipsoids, 2))
        
        self.ellipsoids_centers = centers
        self.ellipsoids_radii = radii

        obj_list = [
            MultiBoxField(np.array(box_centers), np.array(box_sizes), tensor_args=tensor_args),
            MultiEllipsoidField(centers, radii, tensor_args=tensor_args)
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-6.0, -4.5], [6.0, 4.5]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'large_maze_random_ellipsoids')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=25000,
            step_size=0.1,
            n_radius=0.2,
            n_pre_samples=60000,
            max_time=30
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=128,
            n_interpolated_points=None,
            dt=0.08,
            opt_iters=350,
            num_samples=128,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=2e-5,
            step_size=5e-2,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPointMass):
            return params
        raise NotImplementedError

def generate_trajectory(env, robot, task, start_pos, goal_pos, tensor_args, num_trajectories=1, n_support_points=64, gpmp_opt_iters=300, rrt_max_time=15):
    # RRT Star (Informed)
    rrt_connect_params = env.get_rrt_connect_params(robot=robot)
    rrt_connect_params['max_time'] = rrt_max_time
    rrt_params = dict(
        **rrt_connect_params,
        task=task,
        start_state_pos=start_pos,
        goal_state_pos=goal_pos,
        tensor_args=tensor_args,
    )
    rrt_base = InfRRTStar(**rrt_params)
    sample_based_planner = MultiSampleBasedPlanner(
        rrt_base,
        n_trajectories=num_trajectories,
        max_processes=-1,
        optimize_sequentially=True
    )

    # GPMP
    gpmp_params = env.get_gpmp2_params(robot=robot)
    gpmp_params['opt_iters'] = gpmp_opt_iters
    gpmp_params['n_support_points'] = n_support_points
    duration = 5.0
    gpmp_params['dt'] = duration / n_support_points

    gpmp_planner_params = dict(
        **gpmp_params,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_trajectories,
        start_state=start_pos,
        multi_goal_states=goal_pos.unsqueeze(0),
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**gpmp_planner_params)

    hybrid_planner = HybridPlanner(
        sample_based_planner,
        opt_based_planner,
        tensor_args=tensor_args
    )

    trajs_iters = hybrid_planner.optimize(debug=False, print_times=False, return_iterations=True)
    trajs_last_iter = trajs_iters[-1]

    # Final collision check — only return trajectories that are truly collision-free
    _, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)

    success = trajs_last_iter_free is not None and len(trajs_last_iter_free) > 0
    return success, trajs_last_iter_free

if __name__ == '__main__':
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    print("Testing trajectory generation pipeline...")
    env = EnvRandomEllipsoids2D(tensor_args=tensor_args, seed=42)
    robot = RobotPointMass(tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, obstacle_cutoff_margin=0.03, tensor_args=tensor_args)
    
    q_free = task.random_coll_free_q(n_samples=2)
    start_state_pos = q_free[0]
    goal_state_pos = q_free[1]
    
    success, free_trajs = generate_trajectory(env, robot, task, start_state_pos, goal_state_pos, tensor_args)
    print(f"Success: {success}")
    if success:
        print(f"Traj Shape: {free_trajs.shape}")
