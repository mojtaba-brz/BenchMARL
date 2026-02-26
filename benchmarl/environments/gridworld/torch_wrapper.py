import torch
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import make_composite_from_td
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from typing import Any, Optional, Tuple, Dict
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
        super().__init__(device=device)
        self.base_env = base_env
        self.num_agents = num_agents
        
        # Get observation and action specs from base environment
        self._make_specs()
        
        if seed is not None:
            self.set_seed(seed)
    
    def _make_specs(self):
        """Create the environment specs"""
        
        # Get observation shape from base env
        dummy_obs = self.base_env._get_observation()
        obs_shape = dummy_obs[0].shape  # Shape of individual agent observation
        
        # Observation spec (one per agent)
        observation_spec = CompositeSpec({
            f"agent_{i}": BoundedTensorSpec(
                low=0,
                high=2,
                shape=obs_shape,
                dtype=torch.float32,
                device=self.device
            ) for i in range(self.num_agents)
        })
        
        # Global state spec (for critics)
        state = self.base_env.get_state()
        state_spec = BoundedTensorSpec(
            low=0,
            high=2,
            shape=state.shape,
            dtype=torch.float32,
            device=self.device
        )
        
        # Action spec (discrete)
        action_spec = CompositeSpec({
            f"agent_{i}": BoundedTensorSpec(
                low=0,
                high=4,
                shape=(1,),
                dtype=torch.int64,
                device=self.device
            ) for i in range(self.num_agents)
        })
        
        # Reward spec (one per agent)
        reward_spec = CompositeSpec({
            f"agent_{i}": UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device
            ) for i in range(self.num_agents)
        })
        
        # Done spec
        done_spec = CompositeSpec({
            "done": BoundedTensorSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            "terminated": BoundedTensorSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            "truncated": BoundedTensorSpec(
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            )
        })
        
        self.observation_spec = observation_spec
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec
    
    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """Reset the environment"""
        
        # Reset base environment
        obs, info = self.base_env.reset()
        
        # Convert observations to tensor
        obs_tensors = {}
        for i, agent_obs in enumerate(obs):
            obs_tensors[f"agent_{i}"] = torch.tensor(
                agent_obs, dtype=torch.float32, device=self.device
            )
        
        # Get global state
        state = torch.tensor(
            self.base_env.get_state(), dtype=torch.float32, device=self.device
        )
        
        # Create output tensordict
        out = TensorDict(
            {
                **obs_tensors,
                "state": state,
                "done": torch.zeros((1,), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros((1,), dtype=torch.bool, device=self.device),
                "truncated": torch.zeros((1,), dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )
        
        return out
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Take a step in the environment"""
        
        # Extract actions
        actions = []
        for i in range(self.num_agents):
            action_key = f"agent_{i}"
            if action_key in tensordict:
                action = tensordict.get(action_key).item()
            else:
                action = 4  # Default to stay if action missing
            actions.append(action)
        
        # Step base environment
        obs, rewards, done, truncated, info = self.base_env.step(actions)
        
        # Convert observations to tensors
        obs_tensors = {}
        for i, agent_obs in enumerate(obs):
            obs_tensors[f"agent_{i}"] = torch.tensor(
                agent_obs, dtype=torch.float32, device=self.device
            )
        
        # Convert rewards to tensors
        reward_tensors = {}
        for i, reward in enumerate(rewards):
            reward_tensors[f"agent_{i}"] = torch.tensor(
                [reward], dtype=torch.float32, device=self.device
            )
        
        # Get global state
        state = torch.tensor(
            self.base_env.get_state(), dtype=torch.float32, device=self.device
        )
        
        # Create output tensordict
        out = TensorDict(
            {
                **obs_tensors,
                **reward_tensors,
                "state": state,
                "done": torch.tensor([done or truncated], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([done], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )
        
        return out
    
    def set_seed(self, seed: int):
        """Set the random seed"""
        super().set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.base_env.reset(seed=seed)