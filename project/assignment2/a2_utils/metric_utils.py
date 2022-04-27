"""
Lovingly borrowed from https://github.com/dmachlanski/CE888_2022/blob/main/project/metrics.py


"""


import numpy as np
from typing import Tuple, Optional

__all__ = ["abs_ate", "pehe", "abs_att", "policy_risk"]


def abs_ate(effect_true: np.ndarray, effect_pred: np.ndarray) -> float:
    """
    Absolute error for the Average Treatment Effect (ATE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: absolute error on ATE
    """

    return np.abs(np.mean(effect_true) - np.mean(effect_pred))


def pehe(effect_true: np.ndarray, effect_pred: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous treatment Effect (PEHE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: root mean squared error of treatment effect estimations. lower=better.
    """
    return np.sqrt(np.mean(np.square(effect_true - effect_pred)))


def treated_untreated_predicted(
        effect_pred: np.ndarray, yf: np.ndarray, t: np.ndarray, e: Optional[np.ndarray]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    if e is None:
        e = np.ones_like(effect_pred)
    return (
        yf[t > 0],
        yf[(1 - t + e) > 1],
        effect_pred[(t + e) > 1]
    )


def abs_att(effect_pred: np.ndarray, yf: np.ndarray, t: np.ndarray, e: Optional[np.ndarray]) -> float:
    """
    Absolute error for the Average Treatment Effect on the Treated
    :param effect_pred: predicted treatment effect value
    :param yf: factual (observed) outcome
    :param t: treatment status (treated/control)
    :param e: whether belongs to the experimental group. If not given, assume all are in experimental group.
    :return: absolute error on ATT
    """
    if e is None:
        e = np.ones_like(effect_pred)
    att_true = np.mean(yf[t > 0]) - np.mean(yf[(1 - t + e) > 1])
    att_pred = np.mean(effect_pred[(t + e) > 1])

    return np.abs(att_pred - att_true)


def policy_risk(effect_pred: np.ndarray, yf: np.ndarray, t: np.ndarray, e: Optional[np.ndarray]) -> float:
    """
    Computes the risk of the policy defined by predicted effect
    :param effect_pred: predicted treatment effect value
    :param yf: factual (observed) outcome
    :param t: treatment status (treated/control)
    :param e: whether belongs to the experimental group. If not given, assume that entire group is 'experimental'.
    :return: policy risk
    """

    if e is None:
        e = np.ones_like(effect_pred)

    # Consider only the cases for which we have experimental data (i.e., e > 0)
    t_e = t[e > 0]
    yf_e = yf[e > 0]
    effect_pred_e = effect_pred[e > 0]

    if np.any(np.isnan(effect_pred_e)):
        return np.nan

    policy = effect_pred_e > 0.0
    treat_overlap = (policy == t_e) * (t_e > 0)
    control_overlap = (policy == t_e) * (t_e < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(yf_e[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(yf_e[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1.0 - pit) * control_value

    return 1.0 - policy_value


def get_ps_weights(clf, x, t) -> np.ndarray:  # nicked from lab 4 work
    ti = np.squeeze(t)
    clf.fit(x, ti)
    ptx = clf.predict_proba(x).T[1].T + 0.0001  # add a small value to avoid dividing by 0
    # Given ti and ptx values, compute the weights wi (see formula above):
    wi = ti/ptx + ((1-ti)/(1-ptx))
    return wi
