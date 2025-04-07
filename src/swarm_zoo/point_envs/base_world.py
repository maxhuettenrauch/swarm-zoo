from dataclasses import dataclass

import numpy as np

from swarm_zoo.point_envs.agents import PointAgent, AgentState


@dataclass
class WorldParams:
    world_size: tuple[int, int] = (1000., 1000.)
    torus: bool = False


class World:
    def __init__(self, agents: list[PointAgent], world_params: WorldParams = WorldParams()):
        self.agents = agents
        self.size = world_params.world_size
        self.torus = world_params.torus

    def reset(self, agent_states: list[AgentState]):
        for agent, agent_state in zip(self.agents, agent_states):
            agent.reset(agent_state)

    def step(self, actions: list[np.ndarray]):
        for agent, action in zip(self.agents, actions):
            agent.step(action)

        if self.torus:
            # handle toroidal world
            for agent in self.agents:
                agent.state.x = agent.state.x % self.size[0]
                agent.state.y = agent.state.y % self.size[1]
        else:
            # handle bounded world
            for agent in self.agents:
                agent.state.x = np.clip(agent.state.x, 0, self.size[0])
                agent.state.y = np.clip(agent.state.y, 0, self.size[1])