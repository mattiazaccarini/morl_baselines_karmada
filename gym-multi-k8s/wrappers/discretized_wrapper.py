from gymnasium import Wrapper, spaces
import numpy as np

class DiscretizerWrapper(Wrapper):
    def __init__(self, env, n_bins=10):
        super().__init__(env)
        self.n_bins = n_bins
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_shape = self.observation_space.shape

        flat_low = self.obs_low.flatten()
        flat_high = self.obs_high.flatten()
        self.bins = [
            np.linspace(flat_low[i], flat_high[i], self.n_bins + 1)[1:-1]
            for i in range(np.prod(self.obs_shape))
        ]

        self.state_to_index = {}
        self.index_to_state = []
        self.next_index = 0

        self.observation_space = spaces.Discrete(100_000) 

    def discretize_obs(self, obs):
        flat_obs = obs.flatten()
        discretized = tuple(np.digitize(flat_obs[i], self.bins[i]) for i in range(len(flat_obs)))

        if discretized not in self.state_to_index:
            self.state_to_index[discretized] = self.next_index
            self.index_to_state.append(discretized)
            self.next_index += 1

        return self.state_to_index[discretized]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) > 1:
            obs, info = result
            #print(f"Discretized observation: {self.discretize_obs(obs)}")
            return self.discretize_obs(obs), info
        else:
            obs = result
            #print(f"Discretized observation: {self.discretize_obs(obs)}")
            info = {}
            return self.discretize_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        #print(f"Discretized observation: {self.discretize_obs(obs)}")
        #print(f"Discretized observation shape: {self.discretize_obs(obs).shape}")
        #print(f"Discretized observation type: {self.discretize_obs(obs).dtype}")
        #print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        return self.discretize_obs(obs), reward, done, truncated, info
    
    @property
    def spec(self):
        class DummySpec:
            id = "KarmadaSchedulingEnvMulti-v0"
        return DummySpec()
