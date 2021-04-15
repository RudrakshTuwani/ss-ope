from typing import List, Dict

import pandas as pd
import statsmodels.formula.api as smf

from .models import TimestepModel


def construct_Q_model(model_name: str, model_params: Dict) -> TimestepModel:
    if model_name == 'LSTDQ':
        return LSTDQ(formula=model_params["formula"], eval_policy=model_params["eval_policy"])
    else:
        raise NotImplementedError


class LSTDQ(TimestepModel):
    def __init__(self, formula, eval_policy) -> None:
        self.formula = formula
        self.eval_policy = eval_policy

        assert eval_policy.deterministic, "Only works for deterministic policies for now."

    def fit(self, data: List[pd.DataFrame]) -> None:
        num_timesteps = len(data)
        self.models = list()
        pred_future_value = 0

        for t in range(num_timesteps-1, -1, -1):
            df = data[t].copy(deep=True)

            # Target
            df['y'] = df['R'] + pred_future_value

            # Fit model
            model = smf.ols(formula='y ~ ' + self.formula, data=df).fit()
            self.models.append(model)

            # Estimate future value for counterfactual action of eval policy
            df['A'] = self.eval_policy(df)
            pred_future_value = pd.Series(model.predict(df), index=df.index)

        self.models = self.models[::-1]

    def predict(self, data: pd.DataFrame, timestep: int) -> pd.Series:
        Q_SA = pd.Series(self.models[timestep].predict(data), index=data.index)
        return Q_SA
