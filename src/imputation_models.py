from typing import List

import pandas as pd

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold


class ImputationLasso():
    def __init__(self, cv_splits: int, imputation_features: List[str], target: str, seed: int):
        self.cv_splits = cv_splits
        self.imputation_features = imputation_features
        self.seed = seed
        self.target = target

    def find_best_alpha(self, data: pd.DataFrame):
        model = LassoCV(cv=self.cv_splits,
                        random_state=self.seed, normalize=True)
        model.fit(data[self.imputation_features], data[self.target])

        return model.alpha_

    def fit(self, data: List[pd.DataFrame]):
        """ Fits cv_splits number of models to the labeled data and generates cv predictions. """

        num_timesteps = len(data)
        self.models = []
        self.train_residuals = []

        for t in range(num_timesteps):
            df = data[t]

            # Find best alpha for timestep with cross-validation
            alpha = self.find_best_alpha(df)

            # Define cross-validation generator for training models
            cv = KFold(n_splits=self.cv_splits)

            # Placeholders
            pred_target_t = pd.Series(index=df.index)
            cv_models = []

            for train_index, test_index in cv.split(df):

                # Validation data
                val_train = df.iloc[train_index]
                val_test = df.iloc[test_index]

                # Fit model
                model = Lasso(alpha=alpha, normalize=True)
                model.fit(val_train[self.imputation_features],
                          val_train[self.target])

                # Predict
                pred_target_t.iloc[test_index] = model.predict(
                    val_test[self.imputation_features])
                cv_models.append(model)

            # Calculate residuals
            self.train_residuals.append(
                pred_target_t - df[self.target]
            )
            self.models.append(cv_models)

    def predict(self, data: pd.DataFrame, timestep: int):
        pred_target = list()
        for k in range(self.cv_splits):
            model = self.models[timestep][k]
            pred_target.append(
                pd.Series(model.predict(
                    data[self.imputation_features]), index=data.index)
            )
        pred_target = pd.concat(pred_target, axis=1).mean(axis=1)
        return pred_target

    # def predict(self, data: List[pd.DataFrame]):
    #     num_timesteps = len(data)
    #     pred_target = []

    #     for t in range(num_timesteps):

    #         pred_t = list()
    #         for k in range(self.cv_splits):
    #             model = self.models[t][k]
    #             pred_t.append(
    #                 pd.Series(model.predict(
    #                     data[t][self.imputation_features]), index=data[t].index)
    #             )
    #         pred_t = pd.concat(pred_t, axis=1).mean(axis=1)
    #         pred_target.append(pred_t)

    #     return pred_target
