from gymnasium import register


register(
    id='Rendezvous-v0',
    entry_point='swarm_zoo.point_envs:RendezvousEnv',
    max_episode_steps=1000,
    reward_threshold=200.0,
)