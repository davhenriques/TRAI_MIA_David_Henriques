"""Fairness-aware querying (RQ1): ablation and lambda sweep, consensus oracle.

Ablation: random | uncertainty | uncertainty + reweigh | fair λ=2 | fair λ=2 + reweigh
Sweep:    fair + reweigh with λ in {0.5, 1, 2, 4, 8}

Usage:  python scripts/run_fairness.py [--seeds 10]
Output: results/runs/fairness_ablation.csv, results/runs/fairness_sweep.csv,
        results/figures/fairness_*.png
"""

import argparse
import time

import pandas as pd
from scipy.stats import wilcoxon

from trai_al.data import GROUPS
from trai_al.experiment import run_many
from trai_al.metrics import area_under_curve
from trai_al.plots import group_tpr_curves, learning_curves, tradeoff

METRICS = ["balanced_accuracy", "tpr_gap", "worst_group_tpr", "tpr_religion"]
ABLATION = [{"strategy": "random"}, {"strategy": "uncertainty"}, {"strategy": "uncertainty", "reweigh": True},
            {"strategy": "fair", "lam": 2}, {"strategy": "fair", "lam": 2, "reweigh": True}]
LAMBDAS = [0.5, 1, 2, 4, 8]


def auc_table(df):
    return df.groupby(["variant", "seed"]).apply(
        lambda r: pd.Series({m: area_under_curve(r.n_labels, r[m]) for m in METRICS}), include_groups=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = range(args.seeds)

    t0 = time.time()
    abl = run_many(ABLATION, seeds)
    abl.to_csv("results/runs/fairness_ablation.csv", index=False)
    print(f"ablation: {len(ABLATION)} variants x {args.seeds} seeds in {time.time() - t0:.0f}s")

    learning_curves(abl, {"balanced_accuracy": "Balanced accuracy", "tpr_gap": "TPR gap (lower is fairer)",
                          "worst_group_tpr": "Worst-group TPR"}, "results/figures/fairness_curves.png")
    group_tpr_curves(abl[abl.variant.isin(["uncertainty", "fair λ=2 + reweigh"])],
                     "results/figures/fairness_group_tpr.png")

    auc = auc_table(abl)
    order = [c["strategy"] if c["strategy"] != "fair" else f"fair λ={c['lam']:g}" for c in ABLATION]
    order = [o + (" + reweigh" if c.get("reweigh") else "") for o, c in zip(order, ABLATION)]
    summary = auc.groupby("variant").agg(["mean", "std"]).round(3).loc[order]
    print("\nArea under curve (mean ± sd over seeds):")
    print(summary.to_string())

    print("\nPaired Wilcoxon vs uncertainty (difference in area under curve):")
    base = auc.xs("uncertainty", level="variant")
    for v in order:
        if v == "uncertainty":
            continue
        d = auc.xs(v, level="variant") - base
        print(f"  {v:22s} " + "  ".join(f"{m}: {d[m].mean():+.3f} (p={wilcoxon(d[m]).pvalue:.3f})" for m in METRICS))

    final = abl[abl.n_labels == abl.n_labels.max()]
    print("\nAt 550 labels (mean over seeds):")
    print(final.groupby("variant")[["balanced_accuracy", *[f"tpr_{g}" for g in GROUPS], "tpr_gap"]]
          .mean().round(3).loc[order].to_string())
    print("\nHate comments seen per group at 550 labels:")
    print(final.groupby("variant")[[f"seen_hate_{g}" for g in GROUPS]].mean().round(1).loc[order].to_string())

    t0 = time.time()
    sweep = run_many([{"strategy": "fair", "lam": lam, "reweigh": True} for lam in LAMBDAS], seeds)
    sweep.to_csv("results/runs/fairness_sweep.csv", index=False)
    print(f"\nsweep: {len(LAMBDAS)} lambdas x {args.seeds} seeds in {time.time() - t0:.0f}s")
    s_auc = auc_table(sweep).reset_index()
    s_auc["lam"] = s_auc.variant.str.extract(r"λ=([\d.]+)").astype(float)
    g = s_auc.groupby("lam")
    points = pd.DataFrame({"acc": g.balanced_accuracy.mean(), "acc_ci": 1.96 * g.balanced_accuracy.sem(),
                           "gap": g.tpr_gap.mean(), "gap_ci": 1.96 * g.tpr_gap.sem(),
                           "worst": g.worst_group_tpr.mean()}).reset_index()
    print(points.round(3).to_string(index=False))
    ref = {v: (auc.xs(v, level="variant").balanced_accuracy.mean(), auc.xs(v, level="variant").tpr_gap.mean())
           for v in ["random", "uncertainty"]}
    tradeoff(points, "results/figures/fairness_tradeoff.png", ref)


if __name__ == "__main__":
    main()
