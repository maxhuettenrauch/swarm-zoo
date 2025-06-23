import supersuit as ss

from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import SetObsWrapper, LimitedSetObsWrapper, GraphObsWrapper, \
    NearestNeighborGraphObsWrapper


def test_set_obs_wrapper():
    env = RendezvousEnv(num_agents=10, render_mode='human')
    env = SetObsWrapper(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    obs, info = env.reset()
    for i in range(1000):
        actions = [env.action_space.sample() for agent in range(env.num_envs)]
        obs = env.step(actions)
    env.close()

def test_limited_set_obs_wrapper():
    env = RendezvousEnv(num_agents=10, render_mode='human')
    env = LimitedSetObsWrapper(env, max_observed_agents=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    obs, info = env.reset()
    for i in range(1000):
        actions = [env.action_space.sample() for agent in range(env.num_envs)]
        obs = env.step(actions)
    env.close()

def test_graph_obs_wrapper():
    env = RendezvousEnv(num_agents=10, render_mode='human')
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = GraphObsWrapper(env)
    obs, info = env.reset()
    for i in range(1000):
        actions = [env.action_space.sample() for agent in range(env.num_envs)]
        obs = env.step(actions)
    env.close()


def test_nearest_neighbor_graph_obs_wrapper():
    env = RendezvousEnv(num_agents=10, render_mode='human')
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = NearestNeighborGraphObsWrapper(env, max_observed_agents=4)
    obs, info = env.reset()
    for i in range(1000):
        actions = [env.action_space.sample() for agent in range(env.num_envs)]
        obs = env.step(actions)
    env.close()


if __name__ == '__main__':
    test_set_obs_wrapper()
    test_limited_set_obs_wrapper()

    test_graph_obs_wrapper()
    test_nearest_neighbor_graph_obs_wrapper()