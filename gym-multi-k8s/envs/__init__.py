from gymnasium.envs.registration import register

register(
    id='karmada-scheduling-multi-v0',
    entry_point='envs.karmada_scheduling_env_multi:KarmadaSchedulingEnvMulti'
)

register(
    id='karmada-scheduling-multi-v1',
    entry_point='envs.karmada_scheduling_env_multi_power:KarmadaSchedulingEnvMultiPower'
)
