import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.pareto_q_learning.geometric_pql import GeometricPQL
from wrappers.discretized_wrapper import DiscretizerWrapper

import os, pickle, wandb

import numpy as np
import time

num_envs = 4

GAMMA = 0.99
TOTAL_TIMESTEPS = 10000
EVAL_FREQ = 500

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, max_replicas=16, file_results_name="karmada_gym_multi_train_results", is_eval_env=False)
    env = DiscretizerWrapper(env, n_bins=8) # Not required for MOQLearning
    ref_point = np.array([600, 10, 0.9])

    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, max_replicas=16, file_results_name="karmada_gym_multi_eval_results", is_eval_env=True)
    eval_env = DiscretizerWrapper(eval_env, n_bins=8)

    #os.makedirs("checkpoints", exist_ok=True)
    #model_path = ("checkpoints/karmada_morl_pql.pkl")

    agent = PQL(
        env=env,
        ref_point=ref_point,  # used to compute hypervolume
        gamma=GAMMA,
        log=True,  # use weights and biases to see the results!
    )

    agent.train(total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, log_every=1000, action_eval="hypervolume") # Use this for PQL and GeometricPQL
    
    #state_dict = {
    #        "avg_reward":      agent.avg_reward,
    #        "ref_point":       agent.ref_point,
    #        "gamma":           agent.gamma,
    #        "non_dominated":   agent.non_dominated,
    #        "counts":          agent.counts,
    #        "epsilon":         agent.epsilon,
    #        "global_step":     agent.global_step,
    #}
    #with open(model_path, "wb") as f:
    #    pickle.dump(state_dict, f)
    #print("Model saved to:", model_path)
#
    ## Artifact creation for Weights & Biases
    #artifact = wandb.Artifact(
    #    name="karmada_morl_pql",
    #    type="model",
    #    description="Pareto Q-Learning model for Karmada scheduling",
    #)
    #artifact.add_file(model_path)
    ## Log the artifact to Weights & Biases
    #wandb.log_artifact(artifact)
    #wandb.finish()

    