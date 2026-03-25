"""
Utility for extracting circular obstacle definitions from 2D environments.
"""

import sys
import os
from dataclasses import dataclass
from typing import List

# _MPD_DEPS = os.path.join(os.path.dirname(__file__), '..', 'mpd-public', 'deps')

def load_mpd_deps():
    global _MPD_DEPS
    _MPD_DEPS = os.path.join(os.path.abspath(r"C:\Users\Owner\SAFE_DIFFUSION"), 'mpd-public', 'deps')
    print(f'Appending to sys.path: {_MPD_DEPS}')
    sys.path.insert(0, os.path.join(_MPD_DEPS, 'torch_robotics'))
    sys.path.insert(0, os.path.join(_MPD_DEPS, 'storm'))
    sys.path.insert(0, os.path.join(_MPD_DEPS, 'motion_planning_baselines'))
    sys.path.insert(0, os.path.join(_MPD_DEPS, 'experiment_launcher'))


@dataclass
class Circle:
    x: float
    y: float
    r: float

    def __repr__(self):
        return f'Circle(x={self.x:.4f}, y={self.y:.4f}, r={self.r:.4f})'


def get_circles_from_env(env_path: str = None) -> List[Circle]:
    """
    Extract circular obstacles from a 2D environment file.

    Instantiates the environment defined in `env_path`, walks its fixed
    object list, and returns every MultiSphereField entry as a Circle.

    Parameters
    ----------
    env_path : str, optional
        Absolute path to the environment .py file.
        Defaults to EnvSimple2D.

    Returns
    -------
    List[Circle]
        One Circle per sphere in the environment.
    """
    import importlib.util
    import torch
    from torch_robotics.environments.primitives import MultiSphereField

    if env_path is None:
        env_path = os.path.join(
            _MPD_DEPS,
            'torch_robotics', 'torch_robotics', 'environments', 'env_simple_2d.py'
        )

    # Dynamically load the module from path
    spec = importlib.util.spec_from_file_location('_env_module', env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the first EnvBase subclass defined in the module
    from torch_robotics.environments.env_base import EnvBase
    env_class = next(
        cls for name, cls in vars(mod).items()
        if isinstance(cls, type) and issubclass(cls, EnvBase) and cls is not EnvBase
    )

    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    env = env_class(tensor_args=tensor_args, precompute_sdf_obj_fixed=False)

    circles = []
    for obj_field in (env.obj_fixed_list or []):
        for primitive in obj_field.fields:
            if isinstance(primitive, MultiSphereField):
                centers = primitive.centers.cpu().tolist()
                radii = primitive.radii.cpu().tolist()
                for (x, y), r in zip(centers, radii):
                    circles.append(Circle(x=x, y=y, r=r))

    return circles


if __name__ == '__main__':
    circles = get_circles_from_env()
    print(f'{len(circles)} circles found:\n')
    for c in circles:
        print(f'  {c}')
