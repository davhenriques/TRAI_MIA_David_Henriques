"""Pilot study: which base learner suits low-label active learning on Measuring Hate Speech?

Trains each candidate on random labeled subsets of the size range used in the
active learning experiments (50-550 labels) and evaluates on a validation split
carved from the pool (the test set is never touched).

Usage:  python scripts/pilot_model_selection.py
Output: results/pilot_model_selection.csv
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from trai_al.data import GROUPS, load_mhs

warnings.filterwarnings("ignore", category=UserWarning)

SIZES = [50, 100, 300, 550]
N_SEEDS = 10

MODELS = {
    "LogisticRegression (C=1)": lambda s: LogisticRegression(max_iter=2000),
    "LogisticRegression (C=10)": lambda s: LogisticRegression(C=10, max_iter=2000),
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s),
    "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(random_state=s),
    "SVM-RBF (calibrated)": lambda s: CalibratedClassifierCV(SVC(), ensemble=False),
    "kNN (k=15)": lambda s: KNeighborsClassifier(n_neighbors=15),
    "GaussianNB": lambda s: GaussianNB(),
    "MLP (64)": lambda s: MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=s),
}


def group_tpr(y, pred, groups):
    """Share of hate comments detected, per targeted group."""
    return {g: pred[(y == 1) & groups[g].to_numpy()].mean() for g in GROUPS}


def main():
    data = load_mhs(seed=0)
    # validation carved from the pool; stratified by label
    idx_train, idx_val = train_test_split(np.arange(len(data.pool)), test_size=0.2,
                                          stratify=data.pool.y, random_state=0)
    X_tr, y_tr = data.pool.X[idx_train], data.pool.y[idx_train]
    X_val, y_val = data.pool.X[idx_val], data.pool.y[idx_val]
    g_val = data.pool.groups.iloc[idx_val].reset_index(drop=True)

    rows = []
    for name, make in MODELS.items():
        for n in SIZES:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(y_tr), n, replace=False)
                model = make(seed)
                t0 = time.perf_counter()
                model.fit(X_tr[idx], y_tr[idx])
                fit_t = time.perf_counter() - t0
                t0 = time.perf_counter()
                proba_pool = model.predict_proba(X_tr)[:, 1]  # the per-round cost of scoring the pool
                pool_t = time.perf_counter() - t0
                proba = model.predict_proba(X_val)[:, 1]
                pred = (proba >= 0.5).astype(int)
                tpr = group_tpr(y_val, pred, g_val)
                rows.append({
                    "model": name, "n_labels": n, "seed": seed,
                    "accuracy": accuracy_score(y_val, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_val, pred),
                    "brier": brier_score_loss(y_val, proba),
                    **{f"tpr_{g}": v for g, v in tpr.items()},
                    "tpr_gap": max(tpr.values()) - min(tpr.values()),
                    # share of pool instances whose uncertainty score is tied with another one
                    "tied_frac": 1 - len(np.unique(np.round(proba_pool, 10))) / len(proba_pool),
                    "fit_ms": fit_t * 1000, "score_pool_ms": pool_t * 1000,
                })
        print(f"done: {name}")
    df = pd.DataFrame(rows)
    df.to_csv("results/pilot_model_selection.csv", index=False)

    summary = df.groupby(["model", "n_labels"]).agg(
        bal_acc=("balanced_accuracy", "mean"), bal_acc_sd=("balanced_accuracy", "std"),
        brier=("brier", "mean"), tpr_gap=("tpr_gap", "mean"), tpr_gap_sd=("tpr_gap", "std"),
        tied=("tied_frac", "mean"), fit_ms=("fit_ms", "mean"), pool_ms=("score_pool_ms", "mean"),
    ).round(3)
    pd.set_option("display.width", 200)
    print(summary.to_string())
    print("\nTPR per targeted group (mean over seeds):")
    print(df.groupby(["model", "n_labels"])[[f"tpr_{g}" for g in GROUPS]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
