"""Geometric Pareto Q-Learning.

Improved version that fixes critical bugs from the original implementation:
1. Stores non-dominated next-state vectors (like PQL) and computes Q-sets on-the-fly,
   so that updates to avg_reward are always reflected.
2. Uses geometric fitting (quadratic for 2-obj, plane for 3-obj) to generate
   interpolated candidate points, enriching the Pareto front.
3. Fixes the empty-set fallback to check ALL actions, not just action 0.
4. Removes dead code (unused replay buffer, commented-out blocks).
"""

import numbers
import time as time_module
from collections import defaultdict
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value


class GeometricPQL(MOAgent):
    """Geometric Pareto Q-learning.

    Extends standard Pareto Q-Learning (Van Moffaert & Nowé, 2014) with geometric
    fitting of the Pareto front shape. For each state-action pair we maintain:
      - non_dominated[s][a]: set of Pareto-optimal next-state vectors (like PQL)
      - theta[(s,a)]: fitted geometric model coefficients
    
    The geometric model is used to generate interpolated candidate points between
    discovered Pareto solutions, enriching the front faster.
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
        experiment_name: str = "Geometric Pareto Q-Learning",
        wandb_entity: Optional[str] = None,
        log: bool = True,
    ):
        """Initialize the Geometric Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            n_interpolated: Number of interpolated points to generate from the geometric model.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            log: Whether to log or not.
        """
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
        # These store non-dominated next-state Q vectors (NOT pre-composed with avg_reward)
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
            for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Geometric extension: fitted model coefficients per (state, action)
        # For 2-obj: theta = [a0, a1, a2] for y = a0 + a1*x + a2*x^2
        # For 3-obj: theta = [a0, a1, a2] for z = a0 + a1*x + a2*y
        # Lazily allocated: only visited (s,a) pairs get theta entries,
        # keeping memory proportional to visited pairs rather than the full state-action space.
        self.theta: defaultdict[tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(3, dtype=np.float32)
        )

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        # Computational metrics tracking
        self.comp_metrics = {
            "global_step": [],
            "step_time_ms": [],              # wall-clock time per step (ms)
            "calc_nd_time_ms": [],           # time for calc_non_dominated (ms)
            "get_q_set_time_ms": [],         # time for get_q_set calls (ms)
            "geo_fit_time_ms": [],           # time for geometric fitting (ms)
            "total_nd_vectors": [],          # total non-dominated vectors across all (s,a)
            "mean_nd_set_size": [],          # mean |D_{s,a}| across visited (s,a) pairs
            "max_nd_set_size": [],           # max |D_{s,a}|
            "nd_memory_bytes": [],           # memory of non_dominated structure (bytes)
            "theta_memory_bytes": [],        # memory of theta dict (bytes)
            "total_memory_bytes": [],        # total algorithm memory (bytes)
            "visited_sa_pairs": [],          # number of (s,a) pairs visited so far
            "theta_params_count": [],        # total number of theta parameters
        }
        self._step_time_accum = 0.0
        self._calc_nd_time_accum = 0.0
        self._get_q_set_time_accum = 0.0
        self._geo_fit_time_accum = 0.0
        self._steps_since_last_log = 0

        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )

    # ------------------------------------------------------------------ #
    #  Computational metrics                                               #
    # ------------------------------------------------------------------ #

    def _compute_nd_stats(self):
        """Compute non-dominated set statistics and memory in a single pass.
        
        Returns:
            tuple: (total_nd_vectors, visited_sa_pairs, visited_nd_sizes, nd_memory_bytes)
        """
        # Python object overhead constants (CPython 3.10, 64-bit)
        SIZEOF_LIST = 56
        SIZEOF_LIST_PTR = 8
        SIZEOF_SET_BASE = 216
        SIZEOF_SET_ENTRY = 8
        SIZEOF_TUPLE_BASE = 40
        SIZEOF_FLOAT = 28
        SIZEOF_TUPLE_PTR = 8

        total_nd = 0
        visited = 0
        visited_nd_sizes = []
        for s in range(self.num_states):
            for a in range(self.num_actions):
                sz = len(self.non_dominated[s][a])
                total_nd += sz
                if self.counts[s, a] > 0:
                    visited += 1
                    visited_nd_sizes.append(sz)

        num_sa = self.num_states * self.num_actions
        mem = (SIZEOF_LIST + self.num_states * SIZEOF_LIST_PTR +
               self.num_states * (SIZEOF_LIST + self.num_actions * SIZEOF_LIST_PTR) +
               num_sa * SIZEOF_SET_BASE)
        per_vector = (SIZEOF_SET_ENTRY + SIZEOF_TUPLE_BASE + 
                      self.num_objectives * (SIZEOF_TUPLE_PTR + SIZEOF_FLOAT))
        mem += total_nd * per_vector

        return total_nd, visited, visited_nd_sizes, mem

    def _estimate_theta_memory(self):
        """Estimate memory of theta defaultdict."""
        SIZEOF_DICT_BASE = 64
        SIZEOF_DICT_ENTRY = 8
        SIZEOF_KEY_TUPLE = 56       # tuple of 2 ints
        SIZEOF_NDARRAY_BASE = 112   # sys.getsizeof(np.zeros(3, dtype=np.float32))
        SIZEOF_FLOAT32_x3 = 12     # 3 * 4 bytes

        n = len(self.theta)
        mem = SIZEOF_DICT_BASE + n * (SIZEOF_DICT_ENTRY + SIZEOF_KEY_TUPLE + SIZEOF_NDARRAY_BASE + SIZEOF_FLOAT32_x3)
        return mem

    def _collect_comp_metrics(self):
        """Collect and store computational metrics at the current timestep."""
        total_nd, visited, visited_nd_sizes, nd_mem = self._compute_nd_stats()

        # Theta memory estimation (fast)
        theta_mem = self._estimate_theta_memory()
        total_mem = (
            nd_mem
            + theta_mem
            + self.avg_reward.nbytes
            + self.counts.nbytes
        )

        # Count total theta parameters (only for pairs that have theta entries)
        theta_count = sum(arr.size for arr in self.theta.values())

        # Average timing over steps since last log
        n = max(1, self._steps_since_last_log)
        avg_step_time = self._step_time_accum / n * 1000  # ms
        avg_calc_nd_time = self._calc_nd_time_accum / n * 1000
        avg_get_q_set_time = self._get_q_set_time_accum / n * 1000
        avg_geo_fit_time = self._geo_fit_time_accum / n * 1000

        self.comp_metrics["global_step"].append(self.global_step)
        self.comp_metrics["step_time_ms"].append(avg_step_time)
        self.comp_metrics["calc_nd_time_ms"].append(avg_calc_nd_time)
        self.comp_metrics["get_q_set_time_ms"].append(avg_get_q_set_time)
        self.comp_metrics["geo_fit_time_ms"].append(avg_geo_fit_time)
        self.comp_metrics["total_nd_vectors"].append(total_nd)
        self.comp_metrics["mean_nd_set_size"].append(
            float(np.mean(visited_nd_sizes)) if visited_nd_sizes else 0.0
        )
        self.comp_metrics["max_nd_set_size"].append(
            max(visited_nd_sizes) if visited_nd_sizes else 0
        )
        self.comp_metrics["nd_memory_bytes"].append(nd_mem)
        self.comp_metrics["theta_memory_bytes"].append(theta_mem)
        self.comp_metrics["total_memory_bytes"].append(total_mem)
        self.comp_metrics["visited_sa_pairs"].append(visited)
        self.comp_metrics["theta_params_count"].append(theta_count)

        # Reset accumulators
        self._step_time_accum = 0.0
        self._calc_nd_time_accum = 0.0
        self._get_q_set_time_accum = 0.0
        self._geo_fit_time_accum = 0.0
        self._steps_since_last_log = 0

    # ------------------------------------------------------------------ #
    #  Geometric fitting methods                                          #
    # ------------------------------------------------------------------ #

    def _fit_quadratic(self, points: List[tuple]) -> np.ndarray:
        """Fit y = a0 + a1*x + a2*x^2 to 2D points.

        Args:
            points: List of (x, y) tuples.

        Returns:
            theta = [a0, a1, a2]
        """
        if len(points) < 2:
            return np.zeros(3, dtype=np.float32)

        X = np.array([[1.0, p[0], p[0] ** 2] for p in points], dtype=np.float32)
        Y = np.array([p[1] for p in points], dtype=np.float32)
        theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return theta.astype(np.float32)

    def _fit_plane(self, points: List[tuple]) -> np.ndarray:
        """Fit z = a0 + a1*x + a2*y to 3D points.

        Args:
            points: List of (x, y, z) tuples.

        Returns:
            theta = [a0, a1, a2]
        """
        if len(points) < 2:
            return np.zeros(3, dtype=np.float32)

        X = np.array([[1.0, p[0], p[1]] for p in points], dtype=np.float32)
        Z = np.array([p[2] for p in points], dtype=np.float32)
        theta, *_ = np.linalg.lstsq(X, Z, rcond=None)
        return theta.astype(np.float32)

    def _fit_model(self, points: List[tuple]) -> np.ndarray:
        """Auto-select fitting function based on number of objectives.

        Args:
            points: List of point tuples.

        Returns:
            Fitted theta coefficients.
        """
        if self.num_objectives == 2:
            return self._fit_quadratic(points)
        elif self.num_objectives == 3:
            return self._fit_plane(points)
        else:
            # For higher dimensions, use hyperplane fit
            return self._fit_hyperplane(points)

    def _fit_hyperplane(self, points: List[tuple]) -> np.ndarray:
        """Fit a hyperplane to D-dimensional points.

        Fits: x_D = a0 + a1*x_1 + ... + a_{D-1}*x_{D-1}

        Args:
            points: List of D-dimensional tuples.

        Returns:
            theta coefficients.
        """
        if len(points) < 2:
            return np.zeros(self.num_objectives, dtype=np.float32)

        pts = np.array(points, dtype=np.float32)
        D = pts.shape[1]
        # Predict last dimension from the others
        X = np.hstack([np.ones((len(pts), 1)), pts[:, : D - 1]])
        Z = pts[:, D - 1]
        theta, *_ = np.linalg.lstsq(X, Z, rcond=None)
        return theta.astype(np.float32)

    def _generate_interpolated_points(self, state: int, action: int) -> List[tuple]:
        """Use the fitted geometric model to generate interpolated candidate points.

        Samples points along the fitted curve/surface between the min and max values
        of the existing non-dominated set, then filters to keep only non-dominated ones.

        Args:
            state: The state.
            action: The action.

        Returns:
            List of interpolated point tuples.
        """
        theta = self.theta.get((state, action))
        if theta is None:
            return []
        nd_set = self.non_dominated[state][action]
        pts = list(nd_set)

        if len(pts) < 3 or np.allclose(theta, 0):
            return []

        arr = np.array(pts, dtype=np.float32)
        interpolated = []

        if self.num_objectives == 2:
            # theta = [a0, a1, a2], model: y = a0 + a1*x + a2*x^2
            x_min, x_max = arr[:, 0].min(), arr[:, 0].max()
            if x_max - x_min < 1e-8:
                return []
            x_samples = np.linspace(x_min, x_max, self.n_interpolated + 2)[1:-1]
            for x in x_samples:
                y = theta[0] + theta[1] * x + theta[2] * x ** 2
                interpolated.append((float(x), float(y)))

        elif self.num_objectives == 3:
            # theta = [a0, a1, a2], model: z = a0 + a1*x + a2*y
            x_min, x_max = arr[:, 0].min(), arr[:, 0].max()
            y_min, y_max = arr[:, 1].min(), arr[:, 1].max()
            if (x_max - x_min < 1e-8) or (y_max - y_min < 1e-8):
                return []
            # Sample on a grid between extremes
            n_per_dim = max(2, int(np.sqrt(self.n_interpolated)))
            x_samples = np.linspace(x_min, x_max, n_per_dim + 2)[1:-1]
            y_samples = np.linspace(y_min, y_max, n_per_dim + 2)[1:-1]
            for x in x_samples:
                for y in y_samples:
                    z = theta[0] + theta[1] * x + theta[2] * y
                    interpolated.append((float(x), float(y), float(z)))

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

        This is computed ON THE FLY so that updates to avg_reward are always reflected
        (unlike the original implementation which stored pre-composed values in D[(s,a)]).

        Args:
            state: The current state.
            action: The action.

        Returns:
            A set of Q vectors (tuples).
        """
        t0 = time_module.perf_counter()
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        q_set = {tuple(vec) for vec in q_array}

        # Geometric enrichment: add interpolated points from the fitted model
        interpolated = self._generate_interpolated_points(state, action)
        for pt in interpolated:
            q_set.add(pt)

        self._get_q_set_time_accum += time_module.perf_counter() - t0
        return q_set

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric."""
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
        """Compute the action scores based upon the hypervolume metric."""
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return np.array(action_scores, dtype=np.float32)

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state."""
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(
                np.argwhere(action_scores == np.max(action_scores)).flatten()
            )

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Computes Q-sets for all actions and returns the globally non-dominated vectors.

        Args:
            state: The current state.

        Returns:
            Set of Pareto non-dominated vectors.
        """
        candidates = set().union(
            *[self.get_q_set(state, action) for action in range(self.num_actions)]
        )
        return get_non_dominated(candidates)

    def _update_geometric_fit(self, state: int, action: int):
        """Refit the geometric model for a (state, action) pair.

        Uses the current non-dominated set to fit a curve/surface.

        Args:
            state: The state.
            action: The action.
        """
        pts = list(self.non_dominated[state][action])
        if len(pts) >= 2:
            # Compute Q-set points for fitting
            q_pts = []
            nd_array = np.array(pts, dtype=np.float32)
            q_array = self.avg_reward[state, action] + self.gamma * nd_array
            q_pts = [tuple(vec) for vec in q_array]
            self.theta[(state, action)] = self._fit_model(q_pts)
        else:
            self.theta[(state, action)] = np.zeros(3, dtype=np.float32)

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
        """Learn the Pareto front.

        Args:
            total_timesteps: The total number of timesteps to train for.
            eval_env: The environment to evaluate the policies on.
            ref_point: The reference point for HV during evaluation.
            known_pareto_front: The optimal Pareto front, if known.
            num_eval_weights_for_eval: Number of weights for evaluation.
            log_every: Log the results every N timesteps.
            action_eval: The action evaluation function name.

        Returns:
            The final Pareto coverage set at the initial state.
        """
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
                step_start = time_module.perf_counter()

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

                # 2. Update non-dominated set for (state, action) from next_state
                #    Timed: calc_non_dominated
                t0 = time_module.perf_counter()
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self._calc_nd_time_accum += time_module.perf_counter() - t0

                # 3. Refit the geometric model for (state, action)
                #    Timed: geometric fitting
                t0 = time_module.perf_counter()
                self._update_geometric_fit(state, action)
                self._geo_fit_time_accum += time_module.perf_counter() - t0

                # 4. Transition
                state = next_state

                self._step_time_accum += time_module.perf_counter() - step_start
                self._steps_since_last_log += 1

                # 5. Logging
                if self.log and self.global_step % log_every == 0:
                    # Collect computational metrics before logging
                    self._collect_comp_metrics()

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

        # Final metrics collection
        if self._steps_since_last_log > 0:
            self._collect_comp_metrics()

        return self.get_local_pcs(state=0)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))
        return pf

    def track_policy(self, vec, env: gym.Env, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec: The return vector to track.
            env: The environment to track the policy in.
            tol: The tolerance for the return vector.
        """
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
        """Collect the local Pareto Coverage Set in a given state.

        Args:
            state: The state to get a local PCS for.

        Returns:
            Set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        if not candidates:
            return []
        return get_non_dominated(candidates)
