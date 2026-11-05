"""Baselines: random vs uncertainty sampling with a consensus oracle.

Usage:  python scripts/run_baselines.py [--seeds 10]
Output: results/runs/baselines.csv, results/figures/baselines_*.png
"""

import argparse
import time

import pandas as pd

from trai_al.data import GROUPS
from trai_al.experiment import run_many
from trai_al.metrics import area_under_curve
from trai_al.plots import group_tpr_curves, learning_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    df = run_many([{"strategy": "random"}, {"strategy": "uncertainty"}], seeds=range(args.seeds))
    df.to_csv("results/runs/baselines.csv", index=False)
    print(f"{df.seed.nunique()} seeds x {df.strategy.nunique()} strategies in {time.time() - t0:.0f}s")

    learning_curves(df, {"balanced_accuracy": "Balanced accuracy", "tpr_gap": "TPR gap (lower is fairer)",
                         "worst_group_tpr": "Worst-group TPR"}, "results/figures/baselines_curves.png")
    group_tpr_curves(df, "results/figures/baselines_group_tpr.png")

    # area under each curve per run, then mean ± sd across seeds
    metrics = ["balanced_accuracy", "tpr_gap", "worst_group_tpr"]
    auc = df.groupby(["strategy", "seed"]).apply(
        lambda r: pd.Series({m: area_under_curve(r.n_labels, r[m]) for m in metrics}), include_groups=False)
    print("\nArea under curve (mean ± sd over seeds):")
    print(auc.groupby("strategy").agg(["mean", "std"]).round(3).to_string())

    final = df[df.n_labels == df.n_labels.max()]
    print("\nAt 550 labels (mean over seeds):")
    print(final.groupby("strategy")[["balanced_accuracy", *[f"tpr_{g}" for g in GROUPS], "tpr_gap"]]
          .mean().round(3).to_string())
    print("\nHate comments seen during training, per targeted group, at 550 labels:")
    print(final.groupby("strategy")[[f"seen_hate_{g}" for g in GROUPS]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
