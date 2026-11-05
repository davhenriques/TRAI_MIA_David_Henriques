"""RQ2 — accountability: reviewer team (5 personas + 1 stress-test reviewer) via the coordinator.

Conditions (uncertainty sampling throughout):
  oracle            consensus labels (upper reference)
  random            random routing, no quality control
  trust+exclude     v1: trust routing from peer checks + gold; flagged reviewers excluded and remediated
  expert            v2 ablation: + expert adjudication (3/round, escalations first) as the unbiased reference
  expert+posterior  v2: + group-aware noise-corrected (soft) labels

Usage:  python scripts/run_coordinator.py [--seeds 20]
Output: results/runs/coordinator.csv, results/runs/coordinator_trust.csv, results/figures/coordinator_*.png
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from trai_al.data import GROUPS, load_mhs
from trai_al.experiment import Config, make_team, run
from trai_al.metrics import area_under_curve
from trai_al.plots import VARIANT_COLORS, group_tpr_curves, learning_curves

CONDITIONS = ["oracle", "random", "trust+exclude", "expert", "expert+posterior"]
METRICS = ["balanced_accuracy", "tpr_gap", "worst_group_tpr", "tpr_religion", "tpr_gender", "label_error"]


def true_accuracy(team, data):
    """Each reviewer's expected agreement with the consensus over the whole pool."""
    y = data.pool.y
    return {r.rid: float(np.mean(np.where(y == 1, 1, 0) * r.persona.proba(data.pool.score, data.pool.groups)
                                 + np.where(y == 0, 1, 0) * (1 - r.persona.proba(data.pool.score, data.pool.groups))))
            for r in team}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    frames, trust_rows = [], []
    for seed in range(args.seeds):
        data = load_mhs(seed)
        for labeling in CONDITIONS:
            df, _, coord, _ = run(Config(strategy="uncertainty", labeling=labeling, seed=seed), data, return_agents=True)
            frames.append(df)
            if coord is not None:
                acc = true_accuracy(make_team(data, Config(seed=seed), np.random.default_rng(0)), data)
                for rid in coord.reviewers:
                    flag = coord.flags.get(rid)
                    trust_rows.append({"seed": seed, "labeling": labeling, "reviewer": rid, "trust": coord.trust(rid),
                                       "true_accuracy": acc[rid], "flagged": flag is not None,
                                       "flag_group": flag[0] if flag else None, "flag_round": flag[1] if flag else None})
    df = pd.concat(frames, ignore_index=True)
    df.loc[df.labeling == "oracle", "variant"] = "oracle"
    trust = pd.DataFrame(trust_rows)
    df.to_csv("results/runs/coordinator.csv", index=False)
    trust.to_csv("results/runs/coordinator_trust.csv", index=False)
    print(f"{args.seeds} seeds x {len(CONDITIONS)} conditions in {time.time() - t0:.0f}s")

    VARIANT_COLORS.clear()  # this figure's entities, in a fixed order
    VARIANT_COLORS.update(zip(CONDITIONS, ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
    learning_curves(df, {"balanced_accuracy": "Balanced accuracy", "label_error": "Label error (distance from consensus)",
                         "worst_group_tpr": "Worst-group TPR"}, "results/figures/coordinator_curves.png")
    group_tpr_curves(df[df.variant.isin(["oracle", "random", "expert+posterior"])], "results/figures/coordinator_group_tpr.png")

    auc = df.groupby(["variant", "seed"]).apply(
        lambda r: pd.Series({m: area_under_curve(r.n_labels, r[m]) for m in METRICS}), include_groups=False)
    print("\nArea under curve (mean ± sd over seeds):")
    print(auc.groupby("variant").agg(["mean", "std"]).round(3).loc[CONDITIONS].to_string())
    print("\nPaired Wilcoxon vs random routing (difference in area under curve):")
    base = auc.xs("random", level="variant")
    for c in ["trust+exclude", "expert", "expert+posterior"]:
        d = auc.xs(c, level="variant") - base
        print(f"  {c:14s} " + "  ".join(f"{m}: {d[m].mean():+.3f} (p={wilcoxon(d[m]).pvalue:.3f})" for m in METRICS))

    final = df[df.n_labels == df.n_labels.max()]
    print("\nAt 550 labels (mean over seeds):")
    cols = ["balanced_accuracy", "fpr", *[f"tpr_{g}" for g in GROUPS], "tpr_gap", "label_quality", "label_error",
            "labels_spent", "expert_spent"]
    print(final.groupby("variant")[cols].mean().round(3).loc[CONDITIONS].to_string())

    print("\nDetection of the stress-test reviewer (gender blind spot):")
    for c in CONDITIONS[1:]:
        t = trust[trust.labeling == c]
        s, h = t[t.reviewer == "stress test"], t[t.reviewer != "stress test"]
        right_group = (s.flag_group == "gender").sum()
        print(f"  {c:14s} stress flagged {s.flagged.mean():.0%} of seeds (on gender: {right_group}/{len(s)}), "
              f"median round {s.flag_round.median():.0f} | honest reviewers flagged: {h.flagged.sum()} of {len(h)} "
              f"({h[h.flagged].reviewer.value_counts().to_dict()})")

    print("\nTrust validity — Spearman(final trust, true accuracy vs consensus) per seed, honest + stress reviewers:")
    for c in CONDITIONS[1:]:
        t = trust[trust.labeling == c]
        rho = t.groupby("seed").apply(lambda g: spearmanr(g.trust, g.true_accuracy).statistic, include_groups=False)
        print(f"  {c:14s} mean rho = {rho.mean():.2f} (sd {rho.std():.2f})")
    print("\nTrue accuracy vs consensus, per reviewer (seed 0):")
    print(trust[(trust.seed == 0) & (trust.labeling == "expert+posterior")][["reviewer", "true_accuracy", "trust"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
