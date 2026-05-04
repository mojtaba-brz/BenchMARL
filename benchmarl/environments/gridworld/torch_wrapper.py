import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, UnboundedContinuous, Categorical
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from typing import Optional, Dict, List
import numpy as np
import gymnasium as gym

from multiagentcoverage.envs.grid_cpp_env_v0 import GridCPPSimpleEnv
from multiagentcoverage.envs.rewards import reward_cpp_simple
from multiagentcoverage.envs.states import state_fn, state_with_map

class TorchRLGridWorldWrapper(EnvBase):
    """TorchRL wrapper for a general GridWorldEnv"""    
    def __init__(
        self,
        base_env: gym.Env,
        n_agents: int,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        # Set batch_size to empty list for single environment
        super().__init__(device=device, batch_size=())
        self.base_env = base_env
        self.n_agents = n_agents
        self.env_device = device
        
        # Make specs
        self._make_specs()
        
        if seed is not None:
            self.set_seed(seed)
    
    def _make_specs(self):
        """Create the environment specs"""        
        # Create the observation spec using Composite with placeholder specs
        obs, _ = self.base_env.reset()
        states = np.array(obs)
        state_spec = self.base_env.observation_space[0]
        self.observation_spec = Composite(
            agents=Composite(
                state=Bounded(
                    low=state_spec.low[0],
                    high=state_spec.high[0],
                    shape=states.shape,
                    dtype=torch.float32,
                    device=self.device
                ),
                shape=(self.n_agents,)
            )
        )
        
        # Global state spec (for critics)
        self.state_spec = self.observation_spec.clone()
        
        # Action spec
        num_of_actions = self.base_env.action_space[0].n
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(
                    n=num_of_actions-1,
                    shape = (self.n_agents,),
                    dtype = torch.int64, # For compatiblity
                ),
                shape = (self.n_agents,)
            )
        )
        
        # Reward spec 
        self.reward_spec = Composite(
            agents=Composite(
                reward=UnboundedContinuous(
                    shape=torch.Size((self.n_agents, 1)),  # Removed extra dimension
                    device=self.env_device,
                    dtype=torch.float
                ),
                shape=torch.Size((self.n_agents,))
            )
        )
        
        # Done spec
        self.done_spec = Composite(
            done=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
            terminated=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
            truncated=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
        )
    
    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """Reset the environment"""
        
        # Reset base environment
        obs, info = self.base_env.reset()
        initial_state = np.array(obs)
        initial_state = torch.asarray(initial_state, dtype=torch.float32, device=self.device)
        out = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "state": initial_state,
                    },
                    batch_size=torch.Size([self.n_agents]),
                    device=self.device,
                ),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            }
        )
        
        return out
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Take a step in the environment"""
        
        # Step base environment
        actions = tensordict['agents', 'action'].cpu().numpy()
        obs, rewards, done, _, _ = self.base_env.step(actions)
        
        state = torch.asarray(obs, dtype=torch.float32, device=self.device)
        out = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "state": state,
                        "reward": torch.asarray(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),  # Add extra dimension for reward
                    },
                    batch_size=torch.Size([self.n_agents]),
                    device=self.device,
                ),
                "done": torch.asarray([done], dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            }
        )
        
        return out
    
    def _set_seed(self, seed: Optional[int]):
        """Internal method for setting seed"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    # Additional required properties for BenchMARL
    @property
    def group_map(self) -> Dict[str, List[int]]:
        """Return the group map for the environment"""
        return {"agents": list(range(self.n_agents))}
    
    @property
    def has_render(self) -> bool:
        """Return whether the environment has rendering"""
        return False
    
    @property
    def max_steps(self) -> int:
        """Return the maximum number of steps"""
        return self.base_env.max_steps

def test_environment(env:TorchRLGridWorldWrapper):
    td = env.reset()
    reward_sum = 0
    done = False

    while not done:
        td = env.rand_action()
        td = env.step(td)

        rewards = td['next', 'agents', 'reward']
        reward_sum += rewards.mean()
        print(f"Rewards: {rewards.transpose(0, 1)}")
        done = td['next', 'done'][0]

    print("Return: ", reward_sum)
    env.base_env.close()

def test_env_using_benchmarl():
    from benchmarl.algorithms import MappoConfig
    from benchmarl.environments import GridWorldCPPTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.loggers = ["csv"]  # or ["tensorboard"] or []
    experiment_config.max_n_iters = 10

    task = GridWorldCPPTask.DEFAULT.get_from_yaml()

    experiment = Experiment(
    task=task,
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=experiment_config,
    )
    experiment.run()

if __name__ == "__main__":
    # n_agents = 5
    # base_env = GridCPPSimpleEnv(grid_size=10, num_agents=n_agents, spray_capacity=700, max_steps=100, render=True,
    #                             state_fn=state_fn, reward_fn=reward_cpp_simple)
    # env = TorchRLGridWorldWrapper(base_env, n_agents)
    # check_env_specs(env)
    
    # Test basic functionality
    # test_environment(env)
    test_env_using_benchmarl()