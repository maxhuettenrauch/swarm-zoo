from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


@dataclass
class AgentState:
    # physical position and orientation
    x: float
    y: float
    theta: float

    # velocity
    vx: float = 0.0
    vy: float = 0.0

    v_lin: float = 0.0
    v_ang: float = 0.0

    # acceleration
    ax: float = 0.0
    ay: float = 0.0

    a_lin: float = 0.0
    a_ang: float = 0.0

    def get_pos(self):
        return np.array([self.x, self.y])

    def get_w_vel(self):
        return np.array([self.vx, self.vy])

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta,
                         self.vx, self.vy, self.v_lin, self.v_ang,
                         self.ax, self.ay, self.a_lin, self.a_ang])


class PointAgent(ABC):
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.state = AgentState(0, 0, 0)
        self.name = f"agent_{agent_id}"

    def reset(self, agent_state: AgentState):
        self.state = agent_state

    @abstractmethod
    def step(self, action: np.ndarray):
        pass

    @abstractmethod
    def get_observation(self):
        pass


@dataclass
class AgentParams:
    damping: float = 0.0
    # damping factor reducing velocity
    dt: float = 0.01
    # simulation timestep
    n_steps: int = 10
    # number of integration steps
    max_v_lin: float = 10.0
    # max linear velocity
    max_v_ang: float = np.pi
    # max angular velocity


class FirstOrderUnicycleAgent(PointAgent):
    def __init__(self, agent_id: int, agent_params: AgentParams = AgentParams()):
        super().__init__(agent_id)

        self.dt = agent_params.dt
        self.n_steps = agent_params.n_steps
        self.max_v_lin = agent_params.max_v_lin
        self.max_v_ang = agent_params.max_v_ang

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def step(self, action: np.ndarray):
        next_x = self.state.x
        next_y = self.state.y
        next_theta = self.state.theta

        v_lin = action[0] * self.max_v_lin
        v_ang = action[1] * self.max_v_ang

        for i in range(self.n_steps):
            step_x = v_lin * np.cos(self.state.theta)
            step_y = v_lin * np.sin(self.state.theta)

            next_x = next_x + step_x * self.dt
            next_y = next_y + step_y * self.dt
            next_theta = (next_theta + v_ang * self.dt) % (2 * np.pi)

        self.state.x = next_x
        self.state.y = next_y
        self.state.theta = next_theta
        self.state.v_lin = v_lin
        self.state.v_ang = v_ang
        self.state.vx = step_x
        self.state.vy = step_y

    def get_observation(self):
        state = self.state.get_state()
        # only return locally observable state
        # return state[0:3] / 1000.
        local_state = state[[5, 6, 9, 10]]
        # normalize
        return local_state / np.array([self.max_v_lin, self.max_v_ang, 1., 1.])
