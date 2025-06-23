from typing import Tuple

import numpy as np
import numpy.typing as npt


def flat_to_tuple_space(actions_flat: npt.NDArray, num_agents: int, num_envs: int) -> Tuple[npt.NDArray, ...]:
    """
    Convert flat action space to tuple action space representation.

    :param actions_flat: Flat array containing actions for all agents and environments.
                     Shape: (num_envs * num_agents, action_dim)
    :param num_agents: Number of agents
    :param num_envs: Number of environments

    :return: Tuple of arrays where each array contains actions for a single agent across all environments.
            Each element shape: (num_envs, action_dim)
    """
    action_dim = actions_flat.shape[1]
    # Reshape to (num_envs, num_agents, action_dim) and transpose to (num_agents, num_envs, action_dim)
    reshaped = actions_flat.reshape(num_envs, num_agents, action_dim).transpose(1, 0, 2)
    return tuple(reshaped)

def tuple_to_flat_space(actions_tuple: Tuple[npt.NDArray, ...], num_agents: int, num_envs: int) -> npt.NDArray:
    """
    Convert tuple action space to flat action space representation.

    :param actions_tuple: Tuple of arrays where each array contains actions for a single agent across all environments.
                          Each element shape: (num_envs, action_dim)
    :param num_agents: Number of agents
    :param num_envs: Number of environments
    :return: Flat array containing actions for all agents and environments.
             Shape: (num_envs * num_agents, action_dim)
    """

    actions_flat = np.stack(actions_tuple, axis=1).reshape(num_envs * num_agents, -1)
    return actions_flat


def test_flat_to_tuple_space():
    num_agents = 5
    num_envs = 3
    action_dim = 2

    # flat actions contain blocks of actions for each agent in an env,
    # i.e., a_[i + j * num_agents], i in 1,..., num_agents, j in 1, ..., num_envs
    # is agent i's action in the jth env
    actions_flat = np.random.randn(num_envs * num_agents, action_dim)

    # tuple space contains blocks of actions for each agent where blocks are formed per agent,
    # i.e., a[i][j], i and j as above is agent i's action in env j
    actions_tuple = flat_to_tuple_space(actions_flat, num_agents, num_envs)

    for j in range(num_envs):
        for i in range(num_agents):
            assert np.all(actions_flat[j * num_agents + i] == actions_tuple[i][j])

    actions_flat_2 = tuple_to_flat_space(actions_tuple, num_agents, num_envs)

    assert np.all(actions_flat == actions_flat_2)


if __name__ == '__main__':
    test_flat_to_tuple_space()
