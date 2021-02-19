from gym.envs.registration import register

register(
    id='bhs-v1',
    entry_point='bhs.envs:BHSEnv',
)