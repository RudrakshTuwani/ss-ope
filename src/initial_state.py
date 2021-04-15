import pandas as pd
import numpy as np


class InitialState():
    def __init__(self, num_state_features: int, mean: np.array, cov: np.array):
        self.num_state_features = num_state_features
        self.mean = mean
        self.cov = cov

        assert len(mean.shape) == 1
        assert len(cov.shape) == 2

        if mean.shape[0] != num_state_features:
            raise ValueError("Length of the mean vector needs to be the same as the number of state features.")

        if (cov.shape[0] != num_state_features) or (cov.shape[1] != num_state_features):
            raise ValueError("The cov matrix needs to have the same number of rows and columns as the number of state features.")

    def __call__(self, size: int):
        X = np.random.multivariate_normal(self.mean, self.cov, size)
        X = pd.DataFrame(X)
        X.columns = [f'S{d}' for d in range(self.num_state_features)]

        return X
