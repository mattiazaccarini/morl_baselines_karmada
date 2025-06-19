import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning

import numpy as np

num_envs = 4

GAMMA = 0.9
TOTAL_TIMESTEPS = 150000
EVAL_FREQ = 500

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=4, min_replicas=1, max_replicas=16)
    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=4, min_replicas=1, max_replicas=16)
    
    scalarization = tchebicheff(tau=4.0, reward_dim=3)

    agent = MPMOQLearning(
        env,
        learning_rate=0.3,
        scalarization=scalarization,
        use_gpi_policy=False,
        dyna=False,
        initial_epsilon=1,
        final_epsilon=0.01,
        epsilon_decay_steps=int(2e5),
        weight_selection_algo="random",
        epsilon_ols=0.1,
    )

    agent.train(
        total_timesteps=TOTAL_TIMESTEPS, 
        timesteps_per_iteration=15000, 
        eval_freq=100,
        eval_env=eval_env,
        ref_point=np.array([600, 10, 0.9]),
    )