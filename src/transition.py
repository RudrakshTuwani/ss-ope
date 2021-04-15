import patsy
import pandas as pd
import numpy as np


class Transition():
    def __init__(self, formula, weights, cov):
        self.formula = formula
        self.weights = weights
        self.cov = cov

    def __call__(self, data: pd.DataFrame):
        X = patsy.dmatrix(self.formula, data=data, return_type='dataframe')
        X = X @ self.weights
        X = X + data[X.columns]

        return pd.DataFrame(np.concatenate(
            [np.random.multivariate_normal(X.iloc[i], self.cov)[None]
             for i in range(X.shape[0])], 0), columns=X.columns)
