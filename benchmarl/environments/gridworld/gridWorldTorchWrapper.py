import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

# Default values for reset bounds
DEFAULT_X = torch.pi
DEFAULT_Y = 1.0

def angle_normalize(x):
    """Normalize an angle to the range [-pi, pi]."""
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

class PendulumEnv(EnvBase):
    """
    A stateless Pendulum environment inspired by Gymnasium's Pendulum-v1.

    This environment simulates a simple pendulum with torque control. It is stateless,
    meaning the current state (`th`, `thdot`) must be provided in the input tensordict
    at each step, along with system parameters and the action.
    """
    def __init__(self, batch_size=None, device=None):
        if batch_size is None:
            batch_size = torch.Size([])
        super().__init__(device=device, batch_size=batch_size)
        self._make_specs()

    def _make_specs(self):
        """Defines the environment's input and output spaces."""
        # Action space: continuous torque, bounded
        self.action_spec = Bounded(
            shape=torch.Size([1]),
            low=-2.0,
            high=2.0,
            dtype=torch.float32,
            device=self.device,
        )

        # Observation space: composed of angular position (th) and velocity (thdot)
        self.observation_spec = Composite(
            th=Bounded(
                shape=torch.Size([1]),
                low=-torch.pi,
                high=torch.pi,
                dtype=torch.float32,
                device=self.device,
            ),
            thdot=Bounded(
                shape=torch.Size([1]),
                low=-float('inf'),
                high=float('inf'),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # Reward space: unbounded scalar
        self.reward_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        # Done space: boolean flag
        self.done_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

        # State spec: defines the parameters needed for the simulation
        self.state_spec = Composite(
            params=Composite(
                g=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                m=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                l=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                dt=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                max_torque=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                max_speed=Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device),
                shape=torch.Size([]),
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

    def _step(self, tensordict):
        """Executes one step of the pendulum dynamics."""
        # Read state, parameters, and action from the input tensordict
        th, thdot = tensordict["th"], tensordict["thdot"]
        params = tensordict["params"]
        u = tensordict["action"].squeeze(-1)

        # Apply constraints
        u = u.clamp(-params["max_torque"], params["max_torque"])

        # Calculate reward (negative cost)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        reward = -costs.view(*tensordict.shape, 1)

        # Dynamics: update angular velocity and position
        new_thdot = (
            thdot
            + (
                3 * params["g"] / (2 * params["l"]) * th.sin()
                + 3.0 / (params["m"] * params["l"] ** 2) * u
            ) * params["dt"]
        )
        new_thdot = new_thdot.clamp(-params["max_speed"], params["max_speed"])
        new_th = th + new_thdot * params["dt"]

        # Environment is non-terminating
        done = torch.zeros_like(reward, dtype=torch.bool)

        # Prepare output tensordict
        out = TensorDict(
            {
                "th": new_th,
                "thdot": new_thdot,
                "params": params,
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict=None):
        """Resets the environment to a random initial state."""
        if tensordict is None or tensordict.is_empty():
            # Generate default parameters if none are provided
            tensordict = self.gen_params(batch_size=self.batch_size)

        # Define bounds for random initialization
        high_th = torch.tensor(DEFAULT_X, device=self.device)
        high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        # Generate random initial state
        th = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_th - low_th)
            + low_th
        )
        thdot = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_thdot - low_thdot)
            + low_thdot
        )

        # Prepare output tensordict
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _set_seed(self, seed):
        """Sets the seed for the environment's random number generator."""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def gen_params(self, batch_size=None):
        """Generates a set of default parameters for the pendulum simulation."""
        if batch_size is None:
            batch_size = self.batch_size
        return TensorDict(
            {
                "params": {
                    "g": 10.0 * torch.ones(batch_size, 1, device=self.device),
                    "m": 1.0 * torch.ones(batch_size, 1, device=self.device),
                    "l": 1.0 * torch.ones(batch_size, 1, device=self.device),
                    "dt": 0.05 * torch.ones(batch_size, 1, device=self.device),
                    "max_torque": 2.0 * torch.ones(batch_size, 1, device=self.device),
                    "max_speed": 8.0 * torch.ones(batch_size, 1, device=self.device),
                }
            },
            batch_size,
        )