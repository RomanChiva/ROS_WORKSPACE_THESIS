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
class costFn:
    goals = [[20, -5], [20, 5]]
    goal_index: int = 0
    


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    obstacles: ObstaclesConfig
    costfn: costFn
    goal: List[float]
    v_ref: float
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]


cs = ConfigStore.instance()

cs.store(name="ObstacleAvoidance", node=ExampleConfig)
cs.store(name="FreeNav", node=ExampleConfig)
cs.store(name="KL_Cost", node=ExampleConfig)
cs.store(name="Benchmark", node=ExampleConfig)
cs.store(name="config_multi_jackal", node=ExampleConfig)

cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
