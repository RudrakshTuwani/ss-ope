import pandas as pd
import numpy as np


class MDP():
    def __init__(self, P0, R, P, policy, num_actions, num_timesteps, num_state_features):
        self.P0 = P0
        self.R = R
        self.P = P
        self.policy = policy
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
        self.num_state_features = num_state_features

    def sample_trajectories(self, n_trajectories, seed=None):
        return self.sample_trajectories_for_policy(n_trajectories, self.policy, seed)

    def sample_trajectories_random_policy(self, n_trajectories, seed=None):
        def policy(data):
            return [str(a) for a in np.random.randint(0, self.num_actions, size=data.shape[0])]

        return self.sample_trajectories_for_policy(n_trajectories, self.policy, seed)

    def sample_trajectories_for_policy(self, n_trajectories, policy, seed=None):
        if seed is not None:
            np.random.seed(seed)

        data = []
        for t in range(self.num_timesteps):
            data.append(pd.DataFrame())

            # State
            data[t] = self.P0(n_trajectories) if (t == 0) else self.P(data[t-1])

            # Action
            data[t]['A'] = policy(data[t], t)

            # Reward
            data[t]['R'] = self.R(data[t])

        return data
