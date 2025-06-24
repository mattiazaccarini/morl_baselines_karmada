import envs
import gymnasium as gym
import mo_gymnasium as mo_gymnasium
from morl_baselines.multi_policy.envelope.envelope import Envelope
from wrappers.discretized_wrapper import DiscretizerWrapper

import os, pickle, wandb

import numpy as np
import time

GAMMA = 0.99
TOTAL_TIMESTEPS = 10000
EVAL_FREQ = 500

if __name__ == "__main__":
    env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, 
                            min_replicas=1, max_replicas=16, 
                            file_results_name="karmada_gym_multi_train_results_envelope", is_eval_env=False)

    eval_env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, 
                                 max_replicas=16, file_results_name="karmada_gym_multi_eval_results_envelope", is_eval_env=True)

    #os.makedirs("checkpoints", exist_ok=True)
    #model_path = ("checkpoints/karmada_morl_pql.pkl")

    ref_point = np.array([600, 10, 0.9])
    print("Observation space:", env.observation_space)
    env.observation_shape = env.observation_space.shape


    agent = Envelope(
        env=env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=64,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True, # use weights and biases to see the results!
        experiment_name="Envelope-Karmada-MORL",
    )
    
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        num_eval_weights_for_front=100,
        eval_freq=1000,
        log_every=1000,
        action_eval="hypervolume",
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )

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

    