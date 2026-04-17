import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
import pickle, time

import numpy as np

num_envs = 4

GAMMA = 0.9
TOTAL_TIMESTEPS = 20_000
EVAL_FREQ = 1_000

def scalarization_fn(v, w):
    return tchebicheff(tau=4.0, reward_dim=3)(v, w)


if __name__ == "__main__":

    number_of_clusters = [8]
    replicas = [4, 6, 8]

    for num_clusters in number_of_clusters:
        for num_replicas in replicas:

            min_replicas = 1
            max_replicas = num_replicas

            print(f"Training MPMOQLearning with {num_clusters} clusters and {num_replicas} replicas per cluster.")

            env = mo_gymnasium.make("karmada-scheduling-multi-v2", num_clusters=num_clusters,
                                     min_replicas=min_replicas, max_replicas=max_replicas,
                                     file_results_name=f"karmada_gym_c{num_clusters}_r{num_replicas}_results_mpmoq_{time.time()}_nopower",
                                 is_eval_env=False,)
            eval_env = mo_gymnasium.make("karmada-scheduling-multi-v2", num_clusters=num_clusters,
                                      min_replicas=min_replicas, max_replicas=num_replicas,
                                      file_results_name=f"karmada_gym_c{num_clusters}_r{num_replicas}_results_eval_mpmoq_{time.time()}_nopower",
                                      is_eval_env=True)

            ref_point = np.array([
                0.0,
                0.0,
                0.0
            ], dtype=np.float32)

            agent = MPMOQLearning(
                env,
                learning_rate=0.3,
                scalarization=scalarization_fn,
                use_gpi_policy=False,
                dyna=False,
                initial_epsilon=1,
                final_epsilon=0.01,
                epsilon_decay_steps=int(2e5),
                weight_selection_algo="random",
                epsilon_ols=0.1, # not using ols now
                experiment_name=f"MPMOQLearning-Karmada-MORL-C{num_clusters}-R{num_replicas}-NOPower",
            )

            agent.train(
                total_timesteps=TOTAL_TIMESTEPS, 
                timesteps_per_iteration=4000, 
                eval_freq=EVAL_FREQ,
                eval_env=eval_env,
                num_eval_weights_for_front=100,      # ⬅ weights to estimate full Pareto front
                num_eval_episodes_for_front=5,       # ⬅ episodes per weight for full front
                ref_point=ref_point,
            )

