import pickle
import wandb
import os
import envs
import mo_gymnasium
import numpy as np
from wrappers.discretized_wrapper import DiscretizerWrapper


from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

run = wandb.init()
artifact = run.use_artifact('zccmtt/MORL-Baselines/karmada_morl_pql:v1', type='model')
artifact_dir = artifact.download()

model_path = os.path.join(artifact_dir, "karmada_morl_pql.pkl")

env = mo_gymnasium.make("karmada-scheduling-multi-v0", num_clusters=4, min_replicas=1, max_replicas=16)
env = DiscretizerWrapper(env, n_bins=8) # Not required for MOQLearning
      
with open(model_path, "rb") as f:
    state_dict = pickle.load(f)
    print("Model loaded from:", model_path)

agent = PQL(
    env=env,
    ref_point=state_dict["ref_point"],
    gamma=state_dict["gamma"],
    log=True,
)

agent.avg_reward = state_dict["avg_reward"]
agent.non_dominated = state_dict["non_dominated"]
agent.counts = state_dict["counts"]
agent.epsilon = state_dict["epsilon"]
agent.global_step = state_dict["global_step"]

local_pcs = agent.get_local_pcs(state=0)
print("Local PCS (Q-set nondominato) in state 0:", local_pcs)

results = []
for vec in local_pcs:
    for vec in local_pcs:
        rets = [agent.track_policy(vec, env) for _ in range(5)]
        avg_ret = np.mean(rets, axis=0)
        results.append((vec, avg_ret, np.std(rets,axis=0)))
        print(f"Target {vec} → avg return {avg_ret} ± {np.std(rets,axis=0)}")


