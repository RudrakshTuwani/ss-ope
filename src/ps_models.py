from typing import List, Dict

import pandas as pd
import statsmodels.formula.api as smf

from .models import TimestepModel


def construct_PS_model(model_name: str, model_params: Dict) -> TimestepModel:
    if model_name == 'LinearPS':
        return LinearPS(formula=model_params["formula"])
    else:
        raise NotImplementedError


class LinearPS(TimestepModel):
    def __init__(self, formula) -> None:
        self.formula = formula

    def fit(self, data: List[pd.DataFrame], fit_method: str = 'ncg', disp: int = 0) -> None:
        num_timesteps = len(data)
        self.models = list()
        for t in range(0, num_timesteps, 1):
            model = smf.mnlogit('A ~ ' + self.formula,
                                data=data[t].astype({'A': int}))
            model = model.fit(method=fit_method, disp=disp)
            self.models.append(model)

    def predict_prob(self, data: pd.DataFrame, timestep: int) -> pd.DataFrame:
        pred_prob_action = self.models[timestep].predict(data)
        pred_prob_action.columns = [f'A{col}' for col in pred_prob_action]

        return pred_prob_action

    def predict(self, data: pd.DataFrame, timestep: int) -> pd.Series:
        pred_prob_action = self.predict_prob(data, timestep)
        return pred_prob_action.idxmax(axis=1).map(lambda row: row[1:])
