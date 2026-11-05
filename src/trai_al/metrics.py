"""Evaluation metrics.

Performance (accuracy, balanced accuracy) and fairness across targeted groups:
TPR per group = share of hate comments aimed at that group that the model detects;
TPR gap = best-served group's TPR minus worst-served group's TPR (equal opportunity).
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from trai_al.data import GROUPS


def evaluate(learner, split):
    pred = learner.predict(split.X)
    hate = split.y == 1
    tpr = {g: pred[hate & split.groups[g].to_numpy()].mean() for g in GROUPS}
    return {
        "accuracy": accuracy_score(split.y, pred),
        "balanced_accuracy": balanced_accuracy_score(split.y, pred),
        "tpr": pred[hate].mean(),
        "fpr": pred[~hate].mean(),  # false alarms: guards against "fixing" TPR by flagging everything
        **{f"tpr_{g}": v for g, v in tpr.items()},
        "tpr_gap": max(tpr.values()) - min(tpr.values()),
        "worst_group_tpr": min(tpr.values()),
    }


def labeled_composition(labels, split):
    """How many labeled hate comments target each group (what the learner has seen)."""
    idx = np.fromiter(labels, dtype=int)
    hate = split.y[idx] == 1
    groups = split.groups.iloc[idx]
    return {f"seen_hate_{g}": int(groups.loc[hate, g].sum()) for g in GROUPS}


def area_under_curve(n_labels, values):
    """Normalised area under a learning curve: mean value across the label budget."""
    n_labels, values = np.asarray(n_labels, float), np.asarray(values, float)
    return np.trapezoid(values, n_labels) / (n_labels[-1] - n_labels[0])


def gini(values):
    """Inequality of a workload (0 = everyone equal, 1 = one person carries everything)."""
    v = np.sort(np.asarray(values, float))
    if v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))
