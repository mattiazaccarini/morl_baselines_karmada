"""Geometric Pareto Q-Learning (4+ objective variant).

Improved version that fixes critical bugs from the original implementation.
This variant handles 4+ objectives using hyperplane fitting.
See geometric_pql.py for the full list of fixes applied.
"""

import numbers
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value


class GeometricPQL4(MOAgent):
    """Geometric Pareto Q-learning for 4+ objectives.

    Extends standard PQL with hyperplane fitting of the Pareto front.
    For each state-action pair we maintain:
      - non_dominated[s][a]: set of Pareto-optimal next-state vectors (like PQL)
      - theta[(s,a)]: fitted hyperplane model coefficients

    The hyperplane model is used to generate interpolated candidate points
    between discovered Pareto solutions, enriching the front faster.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        n_interpolated: int = 5,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Geometric Pareto Q-Learning (4-obj)",
        wandb_entity: Optional[str] = None,
        log: bool = True,
    ):
        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.n_interpolated = n_interpolated

        # Algorithm setup
        self.ref_point = ref_point

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
            isinstance(self.env.observation_space, gym.spaces.Box)
            and self.env.observation_space.is_bounded(manner="both")
            and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.unwrapped.reward_space.shape[0]
        self.counts = np.zeros((self.num_states, self.num_actions))

        # Core PQL structure: non-dominated sets per (state, action)
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
            for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Geometric extension: fitted hyperplane coefficients per (state, action)
        # theta has D coefficients: a0 + a1*x_1 + ... + a_{D-1}*x_{D-1} ≈ x_D
        self.theta: dict[tuple[int, int], np.ndarray] = {
            (s, a): np.zeros(self.num_objectives, dtype=np.float32)
            for s in range(self.num_states)
            for a in range(self.num_actions)
        }

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )

    # ------------------------------------------------------------------ #
    #  Geometric fitting methods                                          #
    # ------------------------------------------------------------------ #

    def _fit_hyperplane(self, points: List[tuple]) -> np.ndarray:
        """Fit a hyperplane to D-dimensional points.

        Fits: x_D = a0 + a1*x_1 + ... + a_{D-1}*x_{D-1}

        Args:
            points: List of D-dimensional tuples.

        Returns:
            theta coefficients of length D.
        """
        if len(points) < 2:
            return np.zeros(self.num_objectives, dtype=np.float32)

        pts = np.array(points, dtype=np.float32)
        D = pts.shape[1]
        X = np.hstack([np.ones((len(pts), 1), dtype=np.float32), pts[:, : D - 1]])
        Z = pts[:, D - 1]
        theta, *_ = np.linalg.lstsq(X, Z, rcond=None)
        return theta.astype(np.float32)

    def _generate_interpolated_points(self, state: int, action: int) -> List[tuple]:
        """Use the fitted hyperplane to generate interpolated candidate points.

        Samples points along the fitted hyperplane between the min and max values
        of each dimension in the existing non-dominated set.

        Args:
            state: The state.
            action: The action.

        Returns:
            List of interpolated point tuples.
        """
        theta = self.theta[(state, action)]
        nd_set = self.non_dominated[state][action]
        pts = list(nd_set)

        if len(pts) < 3 or np.allclose(theta, 0):
            return []

        arr = np.array(pts, dtype=np.float32)
        D = arr.shape[1]

        # Compute Q-set points for the range
        nd_array = np.array(pts, dtype=np.float32)
        q_array = self.avg_reward[state, action] + self.gamma * nd_array

        # For each free dimension (0..D-2), sample between min and max
        mins = q_array.min(axis=0)
        maxs = q_array.max(axis=0)
        ranges = maxs[:D - 1] - mins[:D - 1]

        if np.any(ranges < 1e-8):
            return []

        # Sample on a grid in the free dimensions
        n_per_dim = max(2, int(self.n_interpolated ** (1.0 / max(1, D - 1))))
        grids = [np.linspace(mins[d], maxs[d], n_per_dim + 2)[1:-1] for d in range(D - 1)]
        mesh = np.meshgrid(*grids, indexing="ij")
        coords = np.stack([m.ravel() for m in mesh], axis=1)  # shape (n_pts, D-1)

        interpolated = []
        for row in coords:
            # theta = [a0, a1, ..., a_{D-1}]
            last_dim = theta[0] + np.dot(theta[1:], row)
            pt = tuple(row.tolist()) + (float(last_dim),)
            interpolated.append(pt)

        return interpolated

    # ------------------------------------------------------------------ #
    #  Core PQL logic (fixed)                                              #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
            "n_interpolated": self.n_interpolated,
        }

    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Q(s,a) = avg_reward(s,a) + gamma * non_dominated(s,a)
        Computed on-the-fly so that avg_reward updates are always reflected.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        q_set = {tuple(vec) for vec in q_array}

        # Geometric enrichment: add interpolated points from the fitted model
        interpolated = self._generate_interpolated_points(state, action)
        for pt in interpolated:
            q_set.add(pt)

        return q_set

    def score_pareto_cardinality(self, state: int):
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, state: int):
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return np.array(action_scores, dtype=np.float32)

    def select_action(self, state: int, score_func: Callable):
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(
                np.argwhere(action_scores == np.max(action_scores)).flatten()
            )

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state."""
        candidates = set().union(
            *[self.get_q_set(state, action) for action in range(self.num_actions)]
        )
        return get_non_dominated(candidates)

    def _update_geometric_fit(self, state: int, action: int):
        """Refit the hyperplane model for a (state, action) pair."""
        pts = list(self.non_dominated[state][action])
        if len(pts) >= 2:
            nd_array = np.array(pts, dtype=np.float32)
            q_array = self.avg_reward[state, action] + self.gamma * nd_array
            q_pts = [tuple(vec) for vec in q_array]
            self.theta[(state, action)] = self._fit_hyperplane(q_pts)
        else:
            self.theta[(state, action)] = np.zeros(self.num_objectives, dtype=np.float32)

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        log_every: Optional[int] = 1000,
        action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto front."""
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")

        if ref_point is None:
            ref_point = self.ref_point

        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "log_every": log_every,
                    "action_eval": action_eval,
                }
            )

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            if not isinstance(state, int):
                state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1

                if not isinstance(next_state, int):
                    next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                # 1. Update visit count and running average reward
                self.counts[state, action] += 1
                self.avg_reward[state, action] += (
                    reward - self.avg_reward[state, action]
                ) / self.counts[state, action]

                # 2. Update non-dominated set from next_state (like PQL)
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)

                # 3. Refit the geometric model
                self._update_geometric_fit(state, action)

                # 4. Transition
                state = next_state

                # 5. Logging
                if self.log and self.global_step % log_every == 0:
                    wandb.log({"global_step": self.global_step})
                    pf = self._eval_all_policies(eval_env)
                    log_all_multi_policy_metrics(
                        current_front=pf,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

            # Decay epsilon at the end of each episode
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_local_pcs(state=0)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))
        return pf

    def track_policy(self, vec, env: gym.Env, tol=1e-3):
        target = np.array(vec)
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            if not isinstance(state, int):
                state = int(np.ravel_multi_index(state, self.env_shape))
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                for q in self.non_dominated[state][action]:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def get_local_pcs(self, state: int = 0):
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        if not candidates:
            return []
        return get_non_dominated(candidates)
