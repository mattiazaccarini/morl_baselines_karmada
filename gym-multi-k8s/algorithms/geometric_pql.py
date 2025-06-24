""" Geometric Pareto Q-Learning."""

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


class GeometricPQL(MOAgent):
    """Initialize the Geometric Pareto Q-learning algorithm.

        For each state-action pair, we maintain 
        - D[(s,a)]: list of non-dominated vectors [(q1, q2), ...]
        - theta[(s,a)]: vector of three coefficients for the polynomial f(x) = a0 + a1x+a2x^2
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
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Pareto Q-Learning",
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

        ''' Instead of using a non-dominated list of sets, we maintain:
        - D[(s,a)]: list of non-dominated point for (s,a)
        - theta[(s,a)]: vector [a0, a1, a2] (quadratic) for our fit
        '''
        self.D: dict[tuple[int, int], List[tuple[float, float, float]]]= {
            (s, a): [] for s in range(self.num_states) for a in range(self.num_actions)
        }
        self.theta: dict[tuple[int, int], np.ndarray] = {
            (s, a): np.zeros(3, dtype=np.float32) for s in range(self.num_states) for a in range(self.num_actions)
        }

        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))
        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        #self.ref_point_dynamic = np.array(ref_point, dtype=np.float32)  # Used to track the reference point for hypervolume computation
        #self.max_values_seen = np.array(ref_point, dtype=np.float32) # Used to track the maximum values seen in the environment to update the ref point
        #self.hv_point_margins = np.array([1.0, 1.0, 0.1], dtype=np.float32)  # Used to compute the hypervolume reference point

        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )


    def _fit_quadratic(self, points: List[tuple[float, float]]) -> np.ndarray:
        '''
        Recieves a list of points and returns theta = [a0, a1, a2] that minimizes
        || Xtheta - Y ||^2 where y = a0 + a1x + a2x^2
        
        '''
        if len(points) == 0:
            print("No points to fit a quadratic function, returning zeros.")
            return np.zeros(3, dtype=np.float32)
        
        X = []
        Y = []
        for x, y in points:
            X.append([1.0, x, x**2])
            Y.append(y)
        
        X = np.array(X, dtype=np.float32) # shape (n, 3)
        Y = np.array(Y, dtype=np.float32) # shape (n,)

        # Solve the least squares problem
        theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return theta.astype(np.float32)
    

    def _fit_plane(self, points: List[tuple[float, float, float]]) -> np.ndarray:
        '''
        Receives a list of points (x, y, z) and finds [a0, a1, a2] that minimizes
        ∑ (a0 + a1*x + a2*y - z)^2.
        '''
        if not points:
            return np.zeros(3, dtype=np.float32)
        
        X = np.array([[1.0, x, y] for (x, y, z) in points], dtype=np.float32)  # shape (n, 3)
        Z = np.array([z for (x, y, z) in points], dtype=np.float32)  # shape (n,)

        theta, *_ = np.linalg.lstsq(X, Z, rcond=None)
        return theta.astype(np.float32)
    
    
    def calc_non_dominated(self, state: int) -> List[tuple]:
        all_pts = []
        for action in range(self.num_actions):
            all_pts.extend(self.D[(state, action)])
        if not all_pts:
            return []

        arr = np.array(all_pts, dtype=np.float32)
        nd = get_non_dominated(arr)  # Get the non-dominated points
        return [tuple(pt) for pt in nd]
    
    def _update_dataset_and_fit(self, state: int, action: int, new_point: tuple[float, float]):
        '''
        1. Check discrete domain: if in D[(state, action)] there is a point that corresponds to new_point, discard it.
        2. If not, insert new_pt into the dataset D[(state, action)].
        3. Remove all points that are dominated by new_point in D[(state, action)].
        4. Fit the new plane on D[(state, action)] and update theta[(state, action)].
        '''
        key = (state, action)
        pts = self.D[key] # Current list of non-dominated points for (state, action)

        # 1. Check if new_point is dominated by any point in D[(state, action)]
    
        # 1. If new point not already present, insert it into the dataset
        if new_point not in pts:
            pts.append(new_point)

        # 2. Remove all points that are dominated by new_point
        arrs = [np.array(pt, dtype=np.float32) for pt in pts]
        nd = get_non_dominated(arrs)  # Get the non-dominated points
        cleaned_pts = []
        for p in nd:
            if isinstance(p, np.ndarray):
                cleaned_pts.append(tuple(p.tolist()))
            else:
                cleaned_pts.append(tuple(p))  # Convert back to tuples

        # 3. Fit the new plane on D[(state, action)] and update theta[(state, action)]
        theta_new = self._fit_plane(cleaned_pts)
        #theta_new = self._fit_quadratic(cleaned_pts)

        self.D[key] = cleaned_pts
        self.theta[key] = theta_new


    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
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
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        scores = []
        for a in range(self.num_actions):
            pts = list(self.get_q_set(state, a))
            if len(pts) == 0:
                scores.append(0.0)
            else:
                scores.append(hypervolume(self.ref_point, [np.array(p) for p in pts]))
        return np.array(scores, dtype=np.float32)

    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        key = (state, action)
        return set(self.D[key])
    
    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())


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
            total_timesteps (int, optional): The number of episodes to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
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
            print(f"Starting new episode at global step {self.global_step}")
            if not isinstance(state, int):
                state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                print(f"Step {self.global_step}: State {state}, Action {action}, Next State {next_state}, Reward {reward}")
                if not isinstance(next_state, int):
                    next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                self.counts[state, action] += 1
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                #print(f"Updated non-dominated set for state {next_state}, action {action}: {self.non_dominated[next_state][action]}")
                self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
                print(f"Updated average reward for state {state}, action {action}: {self.avg_reward[state, action]}")
                # 2. Compute the new vector for (state, action)
                #    Q_new = avg_reward[state,action] + γ * q_next   per ciascun q_next
                # q_next scans the non-dominated points in next state
                # Consider D[(next_state, action)] and use it to compute the new possible Q_new, then filter them via
                # filter_and_update_dataset_and_fit function

                for a in range(self.num_actions):
                    for q_next in self.D[(next_state, a)]:
                        # Compute the new vector
                        q_next = np.array(q_next, dtype=np.float32)
                        q_new = tuple((self.avg_reward[state, action] + self.gamma * q_next).tolist())
                        # Update the dataset and fit the new quadratic
                        self._update_dataset_and_fit(state, action, q_new)

                # If D[(next_state, action)] is empty, we can just add immediate reward
                if len(self.D[next_state, 0]) == 0:
                    q_new_imm = tuple(self.avg_reward[next_state, action].tolist())
                    self._update_dataset_and_fit(state, action, q_new_imm)

                # Next state and logs
                state = next_state
                if self.log and self.global_step % log_every == 0:
                    wandb.log({"global_step": self.global_step})
                    pf = self._eval_all_policies(eval_env)
                    print(f"Evaluated Pareto front at step {self.global_step}: {pf}")
                    log_all_multi_policy_metrics(
                        current_front=pf,
                        hv_ref_point=ref_point,  # Use the static ref point for evaluation
                        #hv_ref_point=self.max_values_seen + self.hv_point_margins,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

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
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
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
                for q in self.D[(state, action)]:
                    q = np.array(q, dtype=np.float32)
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
            print(f"Track policy: Taking action {closest_action} in state {state}, received reward {reward}, new target {new_target}")
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            #self.max_values_seen = np.maximum(self.max_values_seen, total_rew)
            #self.ref_point_dynamic = self.max_values_seen + self.hv_point_margins
            target = new_target

        return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        print(f"Local PCS for state {state}: {candidates}")
        print(f"Non-dominated candidates: {get_non_dominated(candidates)}")
        return get_non_dominated(candidates)


