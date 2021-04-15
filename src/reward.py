import patsy
import pandas as pd
import numpy as np


class Reward():
    def __init__(self, formula, weights):
        self.formula = formula
        self.weights = weights

    def __call__(self, data: pd.DataFrame):
        X = patsy.dmatrix(self.formula, data=data, return_type='dataframe')
        return np.random.normal(loc=X @ self.weights, scale=1)
