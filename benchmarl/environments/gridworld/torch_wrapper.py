import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from typing import Optional, Dict, List
import numpy as np
import gymnasium as gym


class TorchGridSprayWrapper(EnvBase):
    """TorchRL wrapper for GridSprayEnv"""
    
    def __init__(
        self,
        base_env: gym.Env,
        num_agents: int,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        # Set batch_size to empty list for single environment
        super().__init__(device=device, batch_size=[])
        self.base_env = base_env
        self.num_agents = num_agents
        
        # Get observation shape from base env
        dummy_obs = self.base_env._get_observation()
        self.obs_shape = dummy_obs[0].shape
        
        # Make specs
        self._make_specs()
        
        if seed is not None:
            self.set_seed(seed)
    
    def _make_specs(self):
        """Create the environment specs"""
        
        # Create a simple placeholder spec class
        class PlaceholderSpec:
            def __init__(self, shape, dtype, device):
                self.shape = shape
                self.dtype = dtype
                self.device = device
                self.space = None
            
            def __repr__(self):
                return f"PlaceholderSpec(shape={self.shape}, dtype={self.dtype})"
        
        # Create the observation spec using Composite with placeholder specs
        observation_dict = {}
        for i in range(self.num_agents):
            observation_dict[f"agent_{i}"] = PlaceholderSpec(
                shape=self.obs_shape,
                dtype=torch.float32,
                device=self.device
            )
        self.observation_spec = Composite(**observation_dict)
        
        # Global state spec (for critics)
        dummy_state = self.base_env.get_state()
        self.state_spec = PlaceholderSpec(
            shape=dummy_state.shape,
            dtype=torch.float32,
            device=self.device
        )
        
        # Action spec
        action_dict = {}
        for i in range(self.num_agents):
            action_dict[f"agent_{i}"] = PlaceholderSpec(
                shape=(1,),
                dtype=torch.int64,
                device=self.device
            )
        self.action_spec = Composite(**action_dict)
        
        # Action mask spec
        action_mask_dict = {}
        for i in range(self.num_agents):
            action_mask_dict[f"agent_{i}"] = PlaceholderSpec(
                shape=(5,),
                dtype=torch.bool,
                device=self.device
            )
        self.action_mask_spec = Composite(**action_mask_dict)
        
        # Reward spec (one per agent)
        reward_dict = {}
        for i in range(self.num_agents):
            reward_dict[f"agent_{i}"] = PlaceholderSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device
            )
        self.reward_spec = Composite(**reward_dict)
        
        # Done spec
        self.done_spec = Composite(
            done=PlaceholderSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            terminated=PlaceholderSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            truncated=PlaceholderSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            )
        )
        
        # Info spec
        self.info_spec = Composite()
    
    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """Reset the environment"""
        
        # Reset base environment
        obs, info = self.base_env.reset()
        
        # Create output tensordict
        out = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # Add observations
        for i, agent_obs in enumerate(obs):
            out.set(f"agent_{i}", torch.tensor(
                agent_obs, dtype=torch.float32, device=self.device
            ))
        
        # Add global state
        out.set("state", torch.tensor(
            self.base_env.get_state(), dtype=torch.float32, device=self.device
        ))
        
        # Add done flags
        out.set("done", torch.zeros((1,), dtype=torch.bool, device=self.device))
        out.set("terminated", torch.zeros((1,), dtype=torch.bool, device=self.device))
        out.set("truncated", torch.zeros((1,), dtype=torch.bool, device=self.device))
        
        # Add action masks (all actions available by default)
        for i in range(self.num_agents):
            out.set(f"action_mask_{i}", torch.ones((5,), dtype=torch.bool, device=self.device))
        
        # Add empty info
        out.set("info", TensorDict({}, batch_size=self.batch_size, device=self.device))
        
        return out
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Take a step in the environment"""
        
        # Extract actions
        actions = []
        for i in range(self.num_agents):
            action_key = f"agent_{i}"
            if action_key in tensordict.keys():
                action = tensordict.get(action_key).item()
            else:
                action = 4  # Default to stay if action missing
            actions.append(action)
        
        # Step base environment
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        done = terminated or truncated
        
        # Create output tensordict
        out = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # Add next observations
        for i, agent_obs in enumerate(obs):
            out.set(f"agent_{i}", torch.tensor(
                agent_obs, dtype=torch.float32, device=self.device
            ))
        
        # Add rewards
        for i, reward in enumerate(rewards):
            out.set(f"reward_{i}", torch.tensor(
                [reward], dtype=torch.float32, device=self.device
            ))
        
        # Add global state
        out.set("state", torch.tensor(
            self.base_env.get_state(), dtype=torch.float32, device=self.device
        ))
        
        # Add done flags
        out.set("done", torch.tensor([done], dtype=torch.bool, device=self.device))
        out.set("terminated", torch.tensor([terminated], dtype=torch.bool, device=self.device))
        out.set("truncated", torch.tensor([truncated], dtype=torch.bool, device=self.device))
        
        # Add action masks (all actions still available)
        for i in range(self.num_agents):
            out.set(f"action_mask_{i}", torch.ones((5,), dtype=torch.bool, device=self.device))
        
        # Add info
        out.set("info", TensorDict({}, batch_size=self.batch_size, device=self.device))
        
        return out
    
    def set_seed(self, seed: int):
        """Set the random seed"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.base_env.reset(seed=seed)
    
    def _set_seed(self, seed: Optional[int]):
        """Internal method for setting seed"""
        if seed is not None:
            self.set_seed(seed)
    
    # Additional required properties for BenchMARL
    @property
    def group_map(self) -> Dict[str, List[int]]:
        """Return the group map for the environment"""
        return {"agents": list(range(self.num_agents))}
    
    @property
    def has_render(self) -> bool:
        """Return whether the environment has rendering"""
        return False
    
    @property
    def max_steps(self) -> int:
        """Return the maximum number of steps"""
        return self.base_env.max_steps