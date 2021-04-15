import numpy as np
from scipy.special import softmax
import pandas as pd
import patsy


class Policy():
    def __init__(self, formula, weights, deterministic=False):
        self.formula = formula
        self.weights = weights
        self.deterministic = deterministic

    def calculate_action_prob(self, data: pd.DataFrame, timestep: int = None):
        X = patsy.dmatrix(self.formula, data=data, return_type='dataframe')
        scores = X @ self.weights
        probs = softmax(scores.values, axis=1)
        probs = pd.DataFrame(probs, columns=scores.columns, index=scores.index)

        return probs

    def __call__(self, data: pd.DataFrame, timestep: int = None):
        """
        timestep is an optional parameter for compatibility with other policies.
        """

        probs = self.calculate_action_prob(data, timestep)

        if self.deterministic:
            # Remove leading 'A'
            return probs.idxmax(axis=1).map(lambda row: row[1:])
        else:
            A = probs.apply(lambda row: np.argmax(
                np.random.multinomial(n=1, pvals=row.values)), axis=1)
            return A.astype(str)


class ActionValuePolicy():
    def __init__(self, models, num_actions):
        self.models = models
        self.num_actions = num_actions
        self.deterministic = True

    def calculate_action_value(self, data: pd.DataFrame, timestep: int):
        model = self.models[timestep]
        data_copy = data.copy(deep=True)

        pred_value = pd.DataFrame(index=data_copy.index)
        for a in range(self.num_actions):
            data_copy['A'] = str(a)
            pred_value[f'Q_(S,A{a})'] = model.predict(data_copy)

        return pred_value

    def __call__(self, data: pd.DataFrame, timestep: int):
        """
        Action-value function depends on the timestep
        """

        pred_value = self.calculate_action_value(data, timestep)
        return pred_value.idxmax(axis=1).map(lambda row: row[6:-1])
