"""
Sample file to load a saved MPMOQLearning model and restore it.
This code assumes that the model was saved using the MPMOQLearning class
from the morl_baselines.multi_policy.multi_policy_moqlearning module.
"""

import pickle, time, numpy as np
import envs

from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
from morl_baselines.common.scalarization import tchebicheff

import mo_gymnasium, argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MPMOQLearning model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file.")
    return parser.parse_args()

def scalarization_fn(v, w):
    return tchebicheff(tau=4.0, reward_dim=4)(v, w)

def main():
    args = parse_args()
    model_path = args.model_path
    # Create the environment (same as during training)

    env = mo_gymnasium.make("karmada-scheduling-multi-v1",
                             num_clusters=4,
                             min_replicas=1,
                             max_replicas=16,
                             file_results_name=f"karmada_gym_4components_results_eval__mpmoqlearning_{time.time()}",
                             is_eval_env=True)
    
    # Load saved model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Restore into a new agent
    agent = MPMOQLearning(env)  # provide same env and config as during training
    agent.policies = model_data["policies"]

    n_episodes = 25

    for i, policy in enumerate(agent.policies):
        print(f"Evaluating policy {i} {policy} {policy.weights}")
        total_rewards = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = np.zeros(4, dtype=np.float32)

            while not done:
                weights = policy.weights
                action = policy.eval(obs, weights)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            total_rewards.append(ep_reward)

        avg_reward = np.mean(total_rewards, axis=0)
        print(f"Policy {i}: Avg reward over {n_episodes} episodes: {avg_reward}")



if __name__ == "__main__":
    main()

