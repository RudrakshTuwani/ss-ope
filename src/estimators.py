from typing import List

import pandas as pd
import numpy as np

from .models import TimestepModel


def estimate_counterfactual_value(data: pd.DataFrame, timestep: int, Q_model: TimestepModel):
    df = data.copy(deep=True)

    if Q_model.eval_policy.deterministic:

        # Overwrite actual action with counterfactual action
        df['A'] = Q_model.eval_policy(data=df, timestep=timestep)

        # Predict value for counterfactual action
        V_eval_S = Q_model.predict(data=df, timestep=timestep)

    else:
        raise NotImplementedError

    return V_eval_S


def estimate_isr(data: pd.DataFrame, timestep: int, eval_policy, PS_model: TimestepModel):
    df = data.copy(deep=True)

    # Get probability of action taken under eval policy
    df['A'] = eval_policy(data=df, timestep=timestep)

    if eval_policy.deterministic:
        Pr_eval_A = (df['A'] == data['A']).astype(int)
    else:
        raise NotImplementedError

    # Get (estimated) probability of action under behavior policy
    behav_policy_prob_action = PS_model.predict_prob(
        data=data, timestep=timestep)
    behav_policy_prob_action['A'] = data['A']
    Pr_A = behav_policy_prob_action.apply(
        lambda row: row[f'A{row["A"]}'], axis=1)

    # Calculate importance sampling ratio
    rho_t = Pr_eval_A / Pr_A
    return rho_t


def VSUP_DR(data: List[pd.DataFrame], Q_model: TimestepModel, PS_model: TimestepModel) -> float:
    num_timesteps = len(data)
    for t in range(num_timesteps):

        # Estimate value function for eval policy
        data[t]['V_eval_(S)'] = estimate_counterfactual_value(
            data=data[t], timestep=t, Q_model=Q_model)

        # Calculate cumulative importance sampling ratio
        # At timestep
        rho_t = estimate_isr(
            data=data[t], timestep=t, eval_policy=Q_model.eval_policy, PS_model=PS_model)
        # Cumulative
        if t > 0:
            data[t]['rho'] = rho_t * data[t-1]['rho']
        else:
            data[t]['rho'] = rho_t

    VSUP_DR = data[0]['V_eval_(S)'].mean()
    for t in range(num_timesteps):
        if t < (num_timesteps-1):
            VSUP_DR += (data[t]['rho'].clip(0, 1000) * (data[t]['R'] -
                        data[t]['V_eval_(S)'] + data[t+1]['V_eval_(S)'])).mean()
        else:
            VSUP_DR += (data[t]['rho'].clip(0, 1000) *
                        (data[t]['R'] - data[t]['V_eval_(S)'])).mean()

    return VSUP_DR


def VSSL_DR(labeled_data: List[pd.DataFrame], unlabeled_data: List[pd.DataFrame], Q_model: TimestepModel,
            PS_model: TimestepModel, Imputation_model) -> float:
    num_timesteps = len(labeled_data)

    for t in range(num_timesteps):

        for data in [labeled_data, unlabeled_data]:
            # Estimate value function for eval policy
            data[t]['V_eval_(S)'] = estimate_counterfactual_value(
                data=data[t], timestep=t, Q_model=Q_model)

            # Calculate cumulative importance sampling ratio
            # At timestep
            rho_t = estimate_isr(
                data=data[t], timestep=t, eval_policy=Q_model.eval_policy, PS_model=PS_model)
            # Cumulative
            if t > 0:
                data[t]['rho'] = rho_t * data[t-1]['rho']
            else:
                data[t]['rho'] = rho_t

        if Imputation_model.target == 'R':
            # Impute target for unlabeled data
            unlabeled_data[t]['R_hat'] = Imputation_model.predict(
                data=unlabeled_data[t], timestep=t)

            # Calculate misspecification bias
            miss_bias = np.mean(
                labeled_data[t]['rho']*(Imputation_model.train_residuals[t]))

            # Calculate mu_hat
            unlabeled_data[t]['mu_hat'] = (
                unlabeled_data[t]['R_hat']*unlabeled_data[t]['rho']) + miss_bias

        else:
            raise NotImplementedError

    # Compute estimate
    VSSL_DR = unlabeled_data[0]['V_eval_(S)'].mean()
    for t in range(num_timesteps):
        VSSL_DR += unlabeled_data[t]['mu_hat'].mean()

        if t < (num_timesteps-1):
            VSSL_DR -= (unlabeled_data[t]['rho'].clip(0, 1000) *
                        (unlabeled_data[t]['V_eval_(S)'] - unlabeled_data[t+1]['V_eval_(S)'])).mean()

        else:
            VSSL_DR -= (unlabeled_data[t]['rho'].clip(0, 1000) *
                        unlabeled_data[t]['V_eval_(S)']).mean()

    return VSSL_DR
