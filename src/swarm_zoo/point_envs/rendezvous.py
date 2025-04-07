from dataclasses import dataclass
from typing import TypeVar

import gymnasium
import matplotlib

from swarm_zoo.point_envs.agents import FirstOrderUnicycleAgent, AgentState
from swarm_zoo.point_envs.base_world import World
from swarm_zoo.point_envs.utils import get_angle, get_distances, get_distance_matrix

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
# import torch
# from gymnasium.core import WrapperObsType
from gymnasium.spaces import Box, Dict, Graph, GraphInstance
# from gymnasium import ObservationWrapper as GymObservationWrapper
from gymnasium.vector import VectorObservationWrapper
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils.env import AgentID, ActionType, ObsType
# from torch_geometric.data import Data


WrapperObsType = TypeVar("WrapperObsType")


class Rendezvous:
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        agents = [FirstOrderUnicycleAgent(i) for i in range(num_agents)]
        for i, agent in enumerate(agents):
            agent.name = f"agent_{i}"
        self.world = World(agents)
        self.world_size = np.array(self.world.size)

    def reset(self):
        agent_states = np.random.rand(self.num_agents, 3)
        agent_states[:, 0:2] = self.world.size * ((0.55 - 0.45) * agent_states[:, 0:2] + 0.45)
        agent_states[:, 2:3] = 2 * np.pi * agent_states[:, 2:3]

        agent_states_list = [AgentState(*state) for state in agent_states]
        self.world.reset(agent_states_list)

    def get_observation(self) -> dict[AgentID, np.ndarray]:
        return {agent.name: agent.get_observation() for agent in self.world.agents}

    def get_reward(self):
        dist = 0.
        for i, agent in enumerate(self.world.agents[:-1]):
            agent_pos = agent.state.get_pos()
            other_pos = np.array([other.state.get_pos() for j, other in enumerate(self.world.agents[i + 1:])])
            dist += np.sum(get_distances(agent_pos, other_pos, torus=self.world.torus, world_size=self.world_size))

        return -dist / (self.num_agents ** 2 + self.num_agents) * 2



class RendezvousEnv(ParallelEnv[str, np.ndarray, np.ndarray]):
    metadata = {"name": "rendezvous"}

    def __init__(self, num_agents: int = 2, render_mode: str = 'human'):
        self.scenario = Rendezvous(num_agents)

        self.agents = [agent.name for agent in self.scenario.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.scenario.world.agents)
        }

        self.action_spaces = dict(zip(self.agents, [agent.action_space for agent in self.scenario.world.agents]))
        self.observation_spaces = dict(zip(self.agents, [agent.observation_space for agent in self.scenario.world.agents]))

        self.ax = None
        self.render_mode = render_mode

        self._max_episode_steps = 1000
        self._elapsed_steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict[int, np.ndarray], dict[str, dict]]:

        self.scenario.reset()

        obs = self.scenario.get_observation()
        info = {agent: {} for agent in self.agents}

        self.dists = []
        self.angles = []

        self._elapsed_steps = 0

        return obs, info

    def step(self, actions):
        actions_array = [actions[agent] for agent in self.agents]
        self.scenario.world.step(actions_array)
        task_reward = self.scenario.get_reward()
        action_penalty = 0.1 * np.mean([a ** 2 for a in actions_array])
        reward = task_reward - action_penalty

        # add list of dists to agent 1 from all other agents individually
        agent_pos = self.scenario.world.agents[0].state.get_pos()
        other_pos = np.array([other.state.get_pos() for j, other in enumerate(self.scenario.world.agents[1:])])
        self.dists += [get_distances(agent_pos, other_pos, torus=self.scenario.world.torus, world_size=self.scenario.world_size)]
        angles = get_angle(other_pos, agent_pos,
                           torus=self.scenario.world.torus, world_size=self.scenario.world_size
                           ) - self.scenario.world.agents[0].state.theta
        angles_shift = -angles % (2 * np.pi)
        self.angles += [np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)]

        if self.render_mode == "human":
            self.render()

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        else:
            truncated = False

        return (self.scenario.get_observation(),
                {agent: reward for agent in self.agents},
                {agent: False for agent in self.agents},
                {agent: truncated for agent in self.agents},
                {agent: {} for agent in self.agents})

    def render(self) -> None | np.ndarray | str | list:
        if self.ax is None:
            fig, ax = plt.subplots()
            self.ax = ax

        self.ax.clear()
        self.ax.set_aspect('equal')
        # self.ax.set_xlim((0, self.scenario.world_size[0]))
        # self.ax.set_ylim((0, self.scenario.world_size[1]))

        self.ax.set_xlim((400, 600))
        self.ax.set_ylim((400, 600))

        comm_circles = []

        agent_states = np.array([agent.state.get_pos() for agent in self.scenario.world.agents])

        self.ax.scatter(agent_states[:, 0], agent_states[:, 1], c='b', s=10)
        if 11 < 3:
            for i in range(self.num_agents):
                comm_circles.append(plt.Circle((agent_states[i, 0],
                                                    agent_states[i, 1]),
                                                    self.comm_radius, color='g' if i != 0 else 'b', fill=False))

                self.ax.add_artist(comm_circles[i])

        if self.render_mode == 'human':
            plt.pause(0.01)

    def state(self) -> np.ndarray:
        pass


@dataclass
class AgentGraphData:
    adjacency_matrix: np.ndarray
    distance_matrix: np.ndarray
    angle_matrix: np.ndarray
    relative_angle_matrix: np.ndarray
    relative_velocities: np.ndarray

    @classmethod
    def from_env(cls, env: RendezvousEnv):
        world = env.scenario.world
        world_size = env.scenario.world_size

        agent_pos = np.array([agent.state.get_pos() for agent in world.agents])
        agent_theta = np.array([agent.state.theta for agent in world.agents])
        agent_w_vel = np.array([agent.state.get_w_vel() for agent in world.agents])

        # subtract all agents' velocities from the current agent's velocity to obtain relative velocities
        # (check for correct order of axes)
        relative_velocities = np.stack([agent_w_vel - agent.state.get_w_vel() for agent in world.agents], axis=0)

        distance_matrix = get_distance_matrix(agent_pos, torus=world.torus, world_size=world_size, add_to_diagonal=-1)
        adjacency_matrix = np.where((0. < distance_matrix) & (distance_matrix < 10000.0), 1, 0)

        angles = np.vstack([get_angle(agent_pos, agent.state.get_pos(),
                                      torus=world.torus, world_size=world_size) - agent.state.theta for agent in world.agents])

        angles_shift = -angles % (2 * np.pi)
        angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)

        relative_orientation = np.vstack([agent.state.theta - agent_theta for agent in world.agents])
        relative_angle_matrix = np.where(relative_orientation > np.pi, relative_orientation - 2 * np.pi, relative_orientation)

        return cls(adjacency_matrix, distance_matrix, angle_matrix, relative_angle_matrix, relative_velocities)


class ObservationWrapper(wrappers.BaseParallelWrapper):
    env: RendezvousEnv

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class SetObsWrapper(ObservationWrapper):
    def __init__(self, env: RendezvousEnv):
        super().__init__(env)

        self.set_observation_space = Dict({
            'set_obs': Box(low=-np.inf, high=np.inf, shape=(self.env.num_agents - 1, 8)),
            'local_obs': Box(low=-np.inf, high=np.inf, shape=(3, ))
            # 'local_obs': Box(low=-np.inf, high=np.inf, shape=(4, ))
        })

        self.full_action_space = Box(low=-1.0, high=1.0, shape=(2, ))

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.set_observation_space

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.full_action_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        """
        Takes observations of the form {agent: np.ndarray} and returns a modified observation where the graph data is
        grouped by the observation set from the perspective of each agent. This information is stored in the 'set_obs'
        key. The local observation is stored in the 'local_obs' key.

        :param observation:
        :return:
        """
        agent_graph_data = AgentGraphData.from_env(self.env)

        new_obs = {agent: {} for agent in self.env.agents}
        max_observed_agents = np.max(np.sum(agent_graph_data.adjacency_matrix, axis=1))

        # new_obs['set_obs'] = np.zeros((self.env.num_agents, max_observed_agents, 8))
        # new_obs['local_obs'] = np.zeros((self.env.num_agents, 4))

        for i in range(self.env.num_agents):
            # TODO: take care of the case where the agent is not connected to any other agent
            # set_obs = new_obs['set_obs'][i]
            set_obs = np.zeros((max_observed_agents, 8))
            num_observed_agents = np.sum(agent_graph_data.adjacency_matrix[i])
            set_obs[:num_observed_agents, 0] = agent_graph_data.distance_matrix[i][agent_graph_data.adjacency_matrix[i] == 1] / 100
            set_obs[:num_observed_agents, 1] = np.cos(agent_graph_data.angle_matrix[i][agent_graph_data.adjacency_matrix[i] == 1])
            set_obs[:num_observed_agents, 2] = np.sin(agent_graph_data.angle_matrix[i][agent_graph_data.adjacency_matrix[i] == 1])
            set_obs[:num_observed_agents, 3] = np.cos(agent_graph_data.relative_angle_matrix[i][agent_graph_data.adjacency_matrix[i] == 1])
            set_obs[:num_observed_agents, 4] = np.sin(agent_graph_data.relative_angle_matrix[i][agent_graph_data.adjacency_matrix[i] == 1])
            set_obs[:num_observed_agents, 5:7] = agent_graph_data.relative_velocities[i][agent_graph_data.adjacency_matrix[i] == 1] / 20
            set_obs[:num_observed_agents, 7] = 1
            new_obs[self.env.agents[i]]['set_obs'] = set_obs
            new_obs[self.env.agents[i]]['local_obs'] = observation[self.env.agents[i]]
            # new_obs['local_obs'][i] = observation[self.env.agents[i]]

        return new_obs


class LimitedSetObsWrapper(ObservationWrapper):
    def __init__(self, env: RendezvousEnv, max_observed_agents: int):
        super().__init__(env)

        self.max_observed_agents = max_observed_agents

        self.set_observation_space = Dict({
            'set_obs': Box(low=-np.inf, high=np.inf, shape=(max_observed_agents, 8)),
            'local_obs': Box(low=-np.inf, high=np.inf, shape=(4, ))
        })

        self.full_action_space = Box(low=-1.0, high=1.0, shape=(2,))

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.set_observation_space

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.full_action_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        # only observe the max_observed_agents closest agents

        agent_graph_data = AgentGraphData.from_env(self.env)

        new_obs = {agent: {} for agent in self.env.agents}

        for i in range(self.env.num_agents):
            set_obs = np.zeros((self.max_observed_agents, 8))
            num_observed_agents = np.sum(agent_graph_data.adjacency_matrix[i])
            if num_observed_agents > self.max_observed_agents:
                num_observed_agents = self.max_observed_agents

            n_closest_agents_idx = np.argsort(agent_graph_data.distance_matrix[i])[1:num_observed_agents + 1]
            set_obs[:num_observed_agents, 0] = agent_graph_data.distance_matrix[i][n_closest_agents_idx] / 100
            set_obs[:num_observed_agents, 1] = np.cos(agent_graph_data.angle_matrix[i][n_closest_agents_idx])
            set_obs[:num_observed_agents, 2] = np.sin(agent_graph_data.angle_matrix[i][n_closest_agents_idx])
            set_obs[:num_observed_agents, 3] = np.cos(agent_graph_data.relative_angle_matrix[i][n_closest_agents_idx])
            set_obs[:num_observed_agents, 4] = np.sin(agent_graph_data.relative_angle_matrix[i][n_closest_agents_idx])
            set_obs[:num_observed_agents, 5:7] = agent_graph_data.relative_velocities[i][n_closest_agents_idx] / 20
            set_obs[:num_observed_agents, 7] = 1
            new_obs[self.env.agents[i]]['set_obs'] = set_obs
            new_obs[self.env.agents[i]]['local_obs'] = observation[self.env.agents[i]]

        return new_obs


class PyGObsWrapper(VectorObservationWrapper):
    def __init__(self, env: RendezvousEnv):
        super().__init__(env)

        graph_observation_space = Graph(
            # node_space=Box(low=-np.inf, high=np.inf, shape=(3, )),
            node_space=Box(low=-np.inf, high=np.inf, shape=(4, )),
            edge_space=Box(low=-np.inf, high=np.inf, shape=(7, )),
        )

        # self.dict_graph_observation_space = Dict({
        #     'local': Box(low=-np.inf, high=np.inf, shape=(0, )),
        #     'graph': graph_observation_space,
        # })

        self._observation_space = graph_observation_space

        self.full_action_space = Box(low=-1.0, high=1.0, shape=(2, ))

    # def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
    #     return self.dict_graph_observation_space

    def observations(self, obs):

        agent_graph_data = AgentGraphData.from_env(self.env.par_env)

        d_flat = agent_graph_data.distance_matrix[agent_graph_data.adjacency_matrix == 1]
        a_flat = agent_graph_data.angle_matrix[agent_graph_data.adjacency_matrix == 1]
        ra_flat = agent_graph_data.relative_angle_matrix[agent_graph_data.adjacency_matrix == 1]
        rv_flat = agent_graph_data.relative_velocities[agent_graph_data.adjacency_matrix == 1]

        edge_index = np.array(agent_graph_data.adjacency_matrix.nonzero()).T
        edge_attr = np.zeros((edge_index.shape[0], 7))
        edge_attr[:, 0] = d_flat / 100
        edge_attr[:, 1] = np.cos(a_flat)
        edge_attr[:, 2] = np.sin(a_flat)
        edge_attr[:, 3] = np.cos(ra_flat)
        edge_attr[:, 4] = np.sin(ra_flat)
        edge_attr[:, 5:] = rv_flat / 20

        # node_attr = np.array([node_feature for node_feature in observation.values()])
        node_attr = obs

        # edge_index = torch.tensor(agent_graph_data.adjacency_matrix).nonzero().t().contiguous()
        # edge_attr = torch.tensor(np.stack([d_flat, np.cos(a_flat), np.sin(a_flat),
        #                                           np.cos(ra_flat), np.sin(ra_flat)], axis=1))
        # node_attr = torch.tensor([agent.get_observation() for agent in self.env.scenario.world.agents])
        #
        # agent_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        graph = GraphInstance(node_attr, edge_attr, edge_index)

        # new_obs = {agent: {} for agent in self.env.agents}
        # for agent in self.env.agents:
        #     new_obs[agent]['local'] = np.array([])
        # new_obs['graph'] = graph
        # return new_obs
        # TODO: Try returning a pyg Data object, or a dict node_attr, edge_attr, edge_index
        return graph

    def env_is_wrapped(self, wrapper_class):
        return self.env.env_is_wrapped(wrapper_class)


if __name__ == '__main__':
    env = RendezvousEnv(num_agents=4, render_mode='human')
    env = LimitedSetObsWrapper(env, max_observed_agents=2)
    obs, info = env.reset()
    for i in range(1000):
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        actions = {agent: np.array((1, env.action_space(agent).sample()[1])) for agent in env.agents}
        obs = env.step(actions)
    env.close()
