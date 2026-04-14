from __future__ import annotations
from typing import Callable, Dict, List, Optional, Any
import torch
from torchrl.envs import EnvBase, TransformedEnv
from benchmarl.environments import Task, TaskClass
from dataclasses import dataclass, field
import numpy as np

# Import your custom environment
import sys
import os
project_path = os.path.expanduser("~/coverage_path_planning_marl")
sys.path.append(project_path)
from multiagentcoverage.envs.grid_spray_env import GridSprayEnv
from multiagentcoverage.envs.rewards import return_home_reward_fn
from multiagentcoverage.envs.states import state_fn
from benchmarl.utils import DEVICE_TYPING
from benchmarl.environments.gridworld.torch_wrapper import TorchGridSprayWrapper
from torchrl.data import Composite, TensorSpec


@dataclass
class GridSprayConfig:
    """Configuration for GridSpray environment"""
    grid_size: int = 10
    n_agents: int = 4
    spray_capacity: int = 300
    max_steps: int = 100
    coverage_size: int = 1
    
    # This will catch any extra arguments
    def __init__(self, **kwargs):
        # Set default values
        self.grid_size = kwargs.get('grid_size', 10)
        self.n_agents = kwargs.get('n_agents', kwargs.get('num_agents', 4))
        self.spray_capacity = kwargs.get('spray_capacity', 300)
        self.max_steps = kwargs.get('max_steps', 100)
        self.coverage_size = kwargs.get('coverage_size', 1)
        
        # Store any extra args for debugging but don't use them
        self.extra_args = {k: v for k, v in kwargs.items() 
                          if k not in ['grid_size', 'n_agents', 'num_agents', 
                                      'spray_capacity', 'max_steps', 'coverage_size']}
        
        # Print warning about extra args if in debug mode
        if self.extra_args:
            print(f"Warning: GridSprayConfig ignoring extra arguments: {self.extra_args}")


class GridSprayTask(Task):
    """GridSpray environment task for BenchMARL"""
    
    # Define your tasks/variants
    DEFAULT = None  # Will be loaded from YAML
    SMALL_GRID = None
    
    @staticmethod
    def associated_class():
        return GridSprayClass


class GridSprayClass(TaskClass):
    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "gridworld"

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        """Returns a function that creates the environment"""
        
        # Create config from all YAML parameters
        config = GridSprayConfig(**self.config)
        
        def env_fun():
            # Create base gym environment
            base_env = GridSprayEnv(
                render=False,  # Disable rendering for training
                state_fn=state_fn,
                grid_size=config.grid_size,
                num_agents=config.n_agents,  # Use n_agents
                spray_capacity=config.spray_capacity,
                max_steps=config.max_steps,
                coverage_size=config.coverage_size,
                reward_fn=return_home_reward_fn,
                map_file=None
            )
            
            # Wrap in TorchRL environment
            env = TorchGridSprayWrapper(
                base_env=base_env,
                num_agents=config.n_agents,
                device=device
            )
            
            # Apply transformations
            env = TransformedEnv(env)
            env.append_transform(self.get_reward_sum_transform(env))
            for transform in self.get_env_transforms(env):
                env.append_transform(transform)
            
            # Set device and seed
            env = env.to(device)
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                env.set_seed(seed)
            
            return env
        
        return env_fun

    def supports_continuous_actions(self) -> bool:
        """GridSpray uses discrete actions"""
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    # Required abstract method implementations
    def _get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        """Internal method to get environment function"""
        return self.get_env_fun(num_envs, continuous_actions, seed, device)

    def observation_spec(self, env: EnvBase) -> Composite:
        """Return the observation spec"""
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        """Return the action spec"""
        return env.action_spec

    def action_mask_spec(self, env: EnvBase) -> Composite:
        """Return the action mask spec"""
        return env.action_mask_spec

    def state_spec(self, env: EnvBase) -> Optional[TensorSpec]:
        """Return the state spec"""
        return env.state_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Return the info spec"""
        return env.info_spec

    def group_map(self, env: EnvBase) -> Dict[str, List[int]]:
        """Return the group map"""
        return env.group_map

    def max_steps(self, env: EnvBase) -> int:
        """Return the maximum number of steps"""
        return env.max_steps

    def has_render(self, env: EnvBase) -> bool:
        """Return whether the environment can render"""
        return env.has_render