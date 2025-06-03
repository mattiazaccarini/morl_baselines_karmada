#from morl_baselines.multi_policy.moq_learning import MOQLearning
import mo_gymnasium as mo_gymnasium
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from wrappers.discretized_wrapper import DiscretizerWrapper

import numpy as np
import time

num_envs = 4

GAMMA = 0.9
TOTAL_TIMESTEPS = 15_000
EVAL_FREQ = 500

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, max_replicas=16)
    env = DiscretizerWrapper(env, n_bins=8) # Not required for MOQLearning
    

    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, max_replicas=16)
    eval_env = DiscretizerWrapper(eval_env, n_bins=8) # Not required for MOQLearning

    agent = PQL(
    env=env,
    ref_point = np.array([600, 10, 0.9]),  # used to compute hypervolume
    gamma=GAMMA,
    log=True,  # use weights and biases to see the results!
    )

    #agent = MOQLearning(env, weights=np.array([0.33, 0.33, 0.33]))

    #agent.train(total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, start_time=time.time()) # Use this for MOQLearning
    agent.train(total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, ref_point = np.array([600, 10, 0.9]), log_every=250)
    