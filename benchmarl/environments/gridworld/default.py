from dataclasses import dataclass, MISSING
from typing import Any, Optional

@dataclass
class TaskConfig:
    grid_size: int = MISSING
    n_agents: int = MISSING
    spray_capacity: int = MISSING
    max_steps: int = MISSING
    coverage_size: int = MISSING
    reward_fn_name: Any = MISSING
    state_fn_name: Any = MISSING
    map_file: str|None = None