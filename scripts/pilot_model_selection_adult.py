"""Pilot study: which base learner suits low-label active learning on Adult?

Trains each candidate on random labeled subsets of the size range used in the
active learning experiments (50-550 labels) and evaluates on a validation split
carved from the training data (the test set is never touched).
The sensitive attribute (sex) is excluded from the features.

Usage:  python scripts/pilot_model_selection.py
Output: results/pilot_model_selection.csv
"""

import time

import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import equalized_odds_difference
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SIZES = [50, 100, 300, 550]
N_SEEDS = 10

MODELS = {
    "LogisticRegression": lambda s: LogisticRegression(max_iter=2000),
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s),
    "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(random_state=s),
    "SVM-RBF": lambda s: SVC(probability=True, random_state=s),
    "kNN (k=15)": lambda s: KNeighborsClassifier(n_neighbors=15),
    "GaussianNB": lambda s: GaussianNB(),
    "MLP (64)": lambda s: MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=s),
}


def load():
    d = fetch_adult(as_frame=True, data_home="data")
    X = d.data.drop(columns=["fnlwgt", "education"])
    y = (d.target == ">50K").astype(int).values
    sex = X.pop("sex").astype(str).values
    strata = np.char.add(np.char.add(y.astype(str), "_"), sex)
    # test set is held out and ignored here; validation is carved from train
    X_tr, _, y_tr, _, s_tr, _, st_tr, _ = train_test_split(
        X, y, sex, strata, test_size=0.25, stratify=strata, random_state=0)
    X_pool, X_val, y_pool, y_val, _, s_val = train_test_split(
        X_tr, y_tr, s_tr, test_size=0.2, stratify=st_tr, random_state=0)
    cat = X.select_dtypes("category").columns.tolist()
    num = X.select_dtypes("number").columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    # unlabeled pool features are available in pool-based AL, so fitting the encoder on them is legitimate
    return pre.fit_transform(X_pool), y_pool, pre.transform(X_val), y_val, s_val


def main():
    X_pool, y_pool, X_val, y_val, s_val = load()
    rows = []
    for name, make in MODELS.items():
        for n in SIZES:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(y_pool), n, replace=False)
                model = make(seed)
                t0 = time.perf_counter()
                model.fit(X_pool[idx], y_pool[idx])
                fit_t = time.perf_counter() - t0
                t0 = time.perf_counter()
                proba_pool = model.predict_proba(X_pool)[:, 1]  # the per-round cost of scoring the pool
                pool_t = time.perf_counter() - t0
                proba = model.predict_proba(X_val)[:, 1]
                pred = (proba >= 0.5).astype(int)
                rows.append({
                    "model": name, "n_labels": n, "seed": seed,
                    "accuracy": accuracy_score(y_val, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_val, pred),
                    "eo_diff": equalized_odds_difference(y_val, pred, sensitive_features=s_val),
                    "brier": brier_score_loss(y_val, proba),
                    # share of pool instances whose uncertainty score is tied with another one
                    "tied_frac": 1 - len(np.unique(np.round(proba_pool, 10))) / len(proba_pool),
                    "fit_ms": fit_t * 1000, "score_pool_ms": pool_t * 1000,
                })
        print(f"done: {name}")
    df = pd.DataFrame(rows)
    df.to_csv("results/pilot_model_selection.csv", index=False)

    summary = df.groupby(["model", "n_labels"]).agg(
        bal_acc=("balanced_accuracy", "mean"), bal_acc_sd=("balanced_accuracy", "std"),
        acc=("accuracy", "mean"), eo=("eo_diff", "mean"), brier=("brier", "mean"),
        tied=("tied_frac", "mean"), fit_ms=("fit_ms", "mean"), pool_ms=("score_pool_ms", "mean"),
    ).round(3)
    pd.set_option("display.width", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()
