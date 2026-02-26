from __future__ import annotations
from typing import Callable
import torch
from torchrl.envs import EnvBase, TransformedEnv
from benchmarl.environments import Task
from dataclasses import dataclass
import numpy as np

# Import your custom environment
import sys
import os
project_path = os.path.expanduser("~/coverage_path_planning_marl")
sys.path.append(project_path)
from multiagentcoverage.envs.grid_spray_env import GridSprayEnv
from multiagentcoverage.envs.rewards import return_home_reward_fn
from multiagentcoverage.envs.states import state_fn

@dataclass
class GridSprayConfig:
    """Configuration for GridSpray environment"""
    grid_size: int = 10
    num_agents: int = 4
    spray_capacity: int = 300
    max_steps: int = 100
    coverage_size: int = 1


class GridSprayTask(Task):
    """GridSpray environment task for BenchMARL"""
    
    # Define your tasks/variants
    DEFAULT = None  # Will be loaded from YAML
    SMALL_GRID = None

    @classmethod
    def env_name(cls) -> str:
        """Return the environment name (must match folder name in conf/task/)"""
        return "gridspray"

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: str,
    ) -> Callable[[], EnvBase]:
        """Returns a function that creates the environment"""
        
        # Load configuration from YAML
        config = GridSprayConfig(**self.config)
        
        def env_fun():
            # Create base gym environment
            base_env = GridSprayEnv(
                render=False,  # Disable rendering for training
                state_fn=state_fn,
                grid_size=config.grid_size,
                num_agents=config.num_agents,
                spray_capacity=config.spray_capacity,
                max_steps=config.max_steps,
                coverage_size=config.coverage_size,
                reward_fn=return_home_reward_fn,
                map_file=None
            )
            
            # Wrap in TorchRL environment
            env = TorchGridSprayWrapper(
                base_env=base_env,
                num_agents=config.num_agents,
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