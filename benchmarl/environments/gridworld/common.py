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
from multiagentcoverage.envs.rewards import return_home_reward_fn, reward_cpp_simple
from multiagentcoverage.envs.states import state_fn, state_with_map
from benchmarl.utils import DEVICE_TYPING
from benchmarl.environments.gridworld.torchrl_gridworld_wrapper import TorchRLGridWorldWrapper
from torchrl.data import Composite, TensorSpec

from multiagentcoverage.envs import grid_cpp_env_v0

def get_reward_fn_from_str(name):
    if name == 'return_home_reward_fn':
        return return_home_reward_fn
    else:
        return reward_cpp_simple

def get_state_fn_from_str(name):
    if name == 'state_with_map':
        return state_with_map
    else:
        return state_fn



class GridWorldCPPTask(Task):
    DEFAULT = None
    
    @staticmethod
    def associated_class():
        return GridWorldCPPClass


class GridWorldCPPClass(TaskClass):
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
        **kwargs
    ) -> Callable[[], EnvBase]:
        """Returns a function that creates the environment"""
        
        # Create config from all YAML parameters
        config = self.config
        
        if config['map_file'] == 'None':
            config['map_file'] = None
        config['state_fn'] = get_state_fn_from_str(config['state_fn_name'])
        config['reward_fn'] = get_reward_fn_from_str(config['reward_fn_name'])
        
        def env_fun():
            # Create base gym environment
            base_env = GridSprayEnv(
                render=True,
                state_fn=config['state_fn'],
                grid_size=config['grid_size'],
                num_agents=config['n_agents'],
                spray_capacity=config['spray_capacity'],
                max_steps=config['max_steps'],
                coverage_size=config['coverage_size'],
                reward_fn=config['reward_fn'],
                map_file=config['map_file']
            )
            
            # Wrap in TorchRL environment
            env = TorchRLGridWorldWrapper(
                base_env=base_env,
                n_agents=config['n_agents'],
                device=device
            )
            
            # Set device and seed
            env = env.to(device)
            if seed is not None:
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
        return None

    def state_spec(self, env: EnvBase) -> Optional[TensorSpec]:
        """Return the state spec"""
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """Return the info spec"""
        return None

    def group_map(self, env: EnvBase) -> Dict[str, List[int]]:
        """Return the group map"""
        return env.group_map

    def max_steps(self, env: EnvBase) -> int:
        """Return the maximum number of steps"""
        return env.max_steps

    def has_render(self, env: EnvBase) -> bool:
        """Return whether the environment can render"""
        return env.has_render