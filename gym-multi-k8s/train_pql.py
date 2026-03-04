import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from algorithms.pql import PQL
from algorithms.geometric_pql import GeometricPQL
from algorithms.geometric_pql_4_obj import GeometricPQL4
from wrappers.discretized_wrapper import DiscretizerWrapper

import os, pickle, wandb

import numpy as np
import time

GAMMA = 0.99
TOTAL_TIMESTEPS = 10000
EVAL_FREQ = 500

num_clusters = 8  # You can change this to test with different cluster sizes

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=num_clusters, 
                            min_replicas=8, max_replicas=8, 
                            file_results_name="karmada_gym_multi_train_results", is_eval_env=False)
    env = DiscretizerWrapper(env, n_bins=8) # Not required for MOQLearning
    ref_point = np.array([0.0, 0.0, 0.0, 0.0005], dtype=np.float32)  # Reference point for hypervolume calculation

    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v1", num_clusters=num_clusters, min_replicas=8, 
                                 max_replicas=8, file_results_name="karmada_gym_multi_eval_results", is_eval_env=True)
    eval_env = DiscretizerWrapper(eval_env, n_bins=8)

    #os.makedirs("checkpoints", exist_ok=True)
    #model_path = (f"checkpoints/karmada_morl_geometric_pql_{num_clusters}_clusters.pkl")

    agent = PQL(
        env=env,
        ref_point=ref_point,  # used to compute hypervolume
        gamma=GAMMA,
        log=True,  # use weights and biases to see the results!
    )

    #agent = GeometricPQL(
    #    env,
    #    gamma=GAMMA,
    #    ref_point=ref_point,
    #    log=True, # use weights and biases to see the results!
    #)

    #agent = GeometricPQL4(
    #    env,
    #    gamma=GAMMA,
    #    ref_point=ref_point,
    #    log=True, # use weights and biases to see the results!
    #)

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

    