"""Smoke test: verify computational metrics collection and CSV saving.

Runs both PQL and GeometricPQL for a small number of steps with
karmada-scheduling-multi-v0 (3 objectives, no wandb dependency in reset),
log=False to avoid needing wandb credentials.

Checks:
  1. Agents train without errors
  2. comp_metrics dict is populated correctly
  3. CSV files are written and can be read back
"""

import sys
import os
import traceback

import numpy as np
import pandas as pd
import gymnasium

# Add parent dir so envs/ is importable
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch MOAgent.extract_env_info to handle wrapped envs where
# env.observation_space is Discrete (from DiscretizerWrapper) but
# env.unwrapped.observation_space is Box (original env).
from morl_baselines.common.morl_algorithm import MOAgent
_original_extract_env_info = MOAgent.extract_env_info

def _patched_extract_env_info(self, env):
    if env is not None:
        self.env = env
        if isinstance(env.observation_space, gymnasium.spaces.Discrete):
            self.observation_shape = (1,)
            self.observation_dim = env.observation_space.n  # use wrapper's space, not unwrapped
        else:
            self.observation_shape = env.observation_space.shape
            self.observation_dim = env.observation_space.shape[0]

        self.action_space = env.action_space
        if isinstance(env.action_space, (gymnasium.spaces.Discrete, gymnasium.spaces.MultiBinary)):
            self.action_shape = (1,)
            self.action_dim = env.action_space.n
        else:
            self.action_shape = env.action_space.shape
            self.action_dim = env.action_space.shape[0]
        self.reward_dim = env.unwrapped.reward_space.shape[0]

MOAgent.extract_env_info = _patched_extract_env_info

import envs  # noqa: F401  -- registers the karmada environments
import mo_gymnasium
from algorithms.pql import PQL
from algorithms.geometric_pql import GeometricPQL
from wrappers.discretized_wrapper import DiscretizerWrapper


TIMESTEPS = 50       # Very short — just enough to trigger a few steps
LOG_EVERY = 25       # Collect metrics twice during training
NUM_CLUSTERS = 4     # Smaller state space for speed
N_BINS = 4           # Fewer bins => fewer states (4^n_obs_dims)
REPLICAS = 4

OUT_DIR = "./comp_metrics_data"


def make_envs():
    """Create train and eval environments (v0 = 3 objectives)."""
    #env = mo_gymnasium.make(
    #    "karmada-scheduling-multi-v0",
    #    num_clusters=NUM_CLUSTERS,
    #    min_replicas=REPLICAS,
    #    max_replicas=REPLICAS,
    #    file_results_name="smoke_test_train",
    #    is_eval_env=False,
    #)
    #env = DiscretizerWrapper(env, n_bins=N_BINS)
#
    #eval_env = mo_gymnasium.make(
    #    "karmada-scheduling-multi-v0",
    #    num_clusters=NUM_CLUSTERS,
    #    min_replicas=REPLICAS,
    #    max_replicas=REPLICAS,
    #    file_results_name="smoke_test_eval",
    #    is_eval_env=True,
    #)
    #eval_env = DiscretizerWrapper(eval_env, n_bins=N_BINS)
    env = mo_gymnasium.make("deep-sea-treasure-v0")  # Example environment, replace with your actual environment
    eval_env = mo_gymnasium.make("deep-sea-treasure-v0")  # Example evaluation environment, replace with your actual environment
    return env, eval_env


def save_metrics(agent, algo_name):
    """Save comp_metrics to CSV (mirrors train_pql.py logic)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(agent.comp_metrics)
    fname = f"comp_metrics_{algo_name}_{NUM_CLUSTERS}_{REPLICAS}_smoke.csv"
    fpath = os.path.join(OUT_DIR, fname)
    df.to_csv(fpath, index=False)
    return fpath


def validate_metrics(agent, algo_name):
    """Validate that comp_metrics dict is well-formed."""
    m = agent.comp_metrics
    print(f"\n--- {algo_name} comp_metrics validation ---")

    # Check all lists have the same length
    lengths = {k: len(v) for k, v in m.items()}
    print(f"  Metric list lengths: {lengths}")
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) == 1, f"Metric lists have different lengths! {lengths}"
    n = list(unique_lengths)[0]
    assert n > 0, "No metrics were collected!"
    print(f"  OK: {n} data points collected")

    # Check expected keys
    expected_keys = {
        "global_step", "step_time_ms", "calc_nd_time_ms", "get_q_set_time_ms",
        "total_nd_vectors", "mean_nd_set_size", "max_nd_set_size",
        "nd_memory_bytes", "total_memory_bytes", "visited_sa_pairs",
    }
    if algo_name == "GeometricPQL":
        expected_keys |= {"geo_fit_time_ms", "theta_memory_bytes", "theta_params_count"}

    missing = expected_keys - set(m.keys())
    extra = set(m.keys()) - expected_keys
    assert not missing, f"Missing metrics: {missing}"
    if extra:
        print(f"  Note: extra keys (ok): {extra}")
    print(f"  OK: all expected keys present")

    # Check values are reasonable
    for key in m:
        vals = m[key]
        for v in vals:
            assert v is not None, f"{key} contains None!"
            assert not (isinstance(v, float) and np.isnan(v)), f"{key} contains NaN!"
    print(f"  OK: no None/NaN values")

    # Check global_step is increasing
    steps = m["global_step"]
    assert steps == sorted(steps), f"global_step not monotonically increasing: {steps}"
    print(f"  OK: global_step is monotonically increasing: {steps}")

    # Print a sample of values
    print(f"  Last data point:")
    for key in m:
        print(f"    {key}: {m[key][-1]}")

    return True


def test_algorithm(cls, algo_name, ref_point):
    """Run a full smoke test for one algorithm."""
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {algo_name}")
    print(f"{'='*60}")

    env, eval_env = make_envs()
    print(f"  num_states (after discretization): {np.prod(env.observation_space.nvec) if hasattr(env.observation_space, 'nvec') else env.observation_space.shape[0]}")
    print(f"  num_actions: {env.action_space.n if hasattr(env.action_space, 'n') else np.prod(env.action_space.nvec)}")

    agent = cls(
        env=env,
        ref_point=ref_point,
        gamma=0.99,
        log=False,  # No wandb
    )

    print(f"\n  Training for {TIMESTEPS} timesteps (log_every={LOG_EVERY})...")
    agent.train(
        total_timesteps=TIMESTEPS,
        eval_env=eval_env,
        log_every=LOG_EVERY,
        action_eval="hypervolume",
    )
    print(f"  Training complete. global_step = {agent.global_step}")

    # Validate
    validate_metrics(agent, algo_name)

    # Save CSV
    fpath = save_metrics(agent, algo_name)
    print(f"\n  CSV saved to: {fpath}")

    # Read back and verify
    df = pd.read_csv(fpath)
    print(f"  CSV read back: {len(df)} rows, columns: {list(df.columns)}")
    assert len(df) > 0, "CSV is empty!"
    print(f"  OK: CSV is valid")

    env.close()
    eval_env.close()
    return True


def main():
    ref_point_3obj = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ref_point_2obj = np.array([0.0, 0.0], dtype=np.float32)  # For PQL with 2 objectives

    results = {}
    for cls, name in [(PQL, "PQL"), (GeometricPQL, "GeometricPQL")]:
        try:
            results[name] = test_algorithm(cls, name, ref_point_2obj)
        except Exception as e:
            print(f"\n  FAILED: {name}")
            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("  SMOKE TEST RESULTS")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n  All smoke tests passed!")
        return 0
    else:
        print("\n  Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
