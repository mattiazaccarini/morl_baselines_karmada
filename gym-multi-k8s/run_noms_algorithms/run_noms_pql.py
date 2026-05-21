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
TOTAL_TIMESTEPS = 10000
EVAL_FREQ = 1000

if __name__ == "__main__":
    number_of_clusters = [8]
    replicas = [4, 6, 8]

    for num_clusters in number_of_clusters:
        for num_replicas in replicas:
            min_replicas = 1
            max_replicas = num_replicas

            env = mo_gymnasium.make(
                "karmada-scheduling-multi-v1",
                num_clusters=num_clusters,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                file_results_name=(
                    f"karmada_gym_c{num_clusters}_r{num_replicas}_results_pql_{time.time()}"
                ),
                is_eval_env=False,
            )
            env = DiscretizerWrapper(env, n_bins=8)
            ref_point = np.array([0.0, 0.0, 0.0, 0.0005], dtype=np.float32)

            eval_env = mo_gymnasium.make(
                "karmada-scheduling-multi-v1",
                num_clusters=num_clusters,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                file_results_name=(
                    f"karmada_gym_c{num_clusters}_r{num_replicas}_results_eval_pql_{time.time()}"
                ),
                is_eval_env=True,
            )
            eval_env = DiscretizerWrapper(eval_env, n_bins=8)

            agent = PQL(
                env=env,
                ref_point=ref_point,  # Example reference point for hypervolume calculation
                gamma=GAMMA,
                log=True,  # use weights and biases to see the results!
            )

            agent.train(total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, log_every=1000)
