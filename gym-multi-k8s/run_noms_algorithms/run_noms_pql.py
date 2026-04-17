import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from algorithms.pql import PQL
from wrappers.discretized_wrapper import DiscretizerWrapper

import os, wandb

import numpy as np
import time

num_envs = 4
num_clusters = 8  # You can change this to test with different cluster sizes

GAMMA = 0.99
TOTAL_TIMESTEPS = 30000
EVAL_FREQ = 500

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=num_clusters, 
                            min_replicas=8, max_replicas=8, 
                            file_results_name="karmada_gym_multi_train_results", is_eval_env=False)
    env = DiscretizerWrapper(env, n_bins=8) 
    ref_point = np.array([0.0, 0.0, 0.0, 0.0005], dtype=np.float32)  # Reference point for hypervolume calculation

    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=num_clusters, min_replicas=8, 
                                 max_replicas=8, file_results_name="karmada_gym_multi_eval_results", is_eval_env=True)
    eval_env = DiscretizerWrapper(eval_env, n_bins=8)


    agent = PQL(
        env=env,
        ref_point=ref_point,  # Example reference point for hypervolume calculation
        gamma=GAMMA,
        log=True,  # use weights and biases to see the results!
    )

    agent.train(total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, log_every=5000)
