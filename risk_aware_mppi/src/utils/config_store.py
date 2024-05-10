from dataclasses import dataclass, field
from planner.mppi import MPPIConfig
from hydra.core.config_store import ConfigStore

from typing import List, Optional


@dataclass
class ObstaclesConfig:
    num_obstacles: int = 10
    cov_growth_factor: float = 1.05
    max_velocity: float = 4
    init_area: float = 6
    init_bias: float = 2
    initial_covariance: float = 0.03
    print_time: bool = False
    use_gaussian_batch: bool = True
    N_monte_carlo: int = 20000
    sample_bound: int = 5
    integral_radius: float = 0.15
    split_calculation: bool = False


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    obstacles: ObstaclesConfig
    goal: List[float]
    v_ref: float
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]


cs = ConfigStore.instance()
cs.store(name="config_point_robot", node=ExampleConfig)
cs.store(name="config_multi_point_robot", node=ExampleConfig)
cs.store(name="config_heijn_robot", node=ExampleConfig)
cs.store(name="config_boxer_robot", node=ExampleConfig)
cs.store(name="config_jackal_robot", node=ExampleConfig)
cs.store(name="config_multi_jackal", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(name="config_omnipanda", node=ExampleConfig)
cs.store(name="config_panda_push", node=ExampleConfig)
cs.store(name="config_heijn_push", node=ExampleConfig)
cs.store(name="config_boxer_push", node=ExampleConfig)
cs.store(name="config_panda_c_space_goal", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
