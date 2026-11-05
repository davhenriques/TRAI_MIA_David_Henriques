"""Headline comparison: the final system vs naive human-in-the-loop vs the consensus oracle.

  oracle        consensus labels — what perfect annotation would give (upper reference)
  naive HITL    one reviewer per comment, random routing, no quality control
  final system  coordinator v2 (expert references + group-aware corrected labels) + supervisor
                (human-decided exclusions, calibrated report) + exposure cap 1.2 + monitoring
                (expert budget: 2 adjudications + 1 random monitoring item per round)

Usage:  python scripts/run_final.py [--seeds 20]
Output: results/runs/final.csv, results/tables/final.md, results/figures/final_*.png
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from trai_al.data import GROUPS
from trai_al.experiment import run_many
from trai_al.metrics import area_under_curve
from trai_al.plots import VARIANT_COLORS, group_tpr_curves, learning_curves

FINAL = dict(labeling="expert+posterior, flags only", supervisor=True, exposure_cap=1.2,
             expert_per_round=2, monitor_per_round=1)
CONDITIONS = {"oracle": dict(labeling="oracle"), "naive HITL": dict(labeling="random"), "final system": FINAL}
METRICS = ["balanced_accuracy", "fpr", "tpr_gap", "worst_group_tpr"]


def format_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    frames = []
    for name, c in CONDITIONS.items():
        d = run_many([{"strategy": "uncertainty", **c}], range(args.seeds))
        d["variant"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("results/runs/final.csv", index=False)

    VARIANT_COLORS.clear()
    VARIANT_COLORS.update(zip(CONDITIONS, ["#52514e", "#2a78d6", "#1baf7a"]))
    learning_curves(df, {"balanced_accuracy": "Balanced accuracy", "tpr_gap": "TPR gap (lower is fairer)",
                         "worst_group_tpr": "Worst-group TPR"}, "results/figures/final_curves.png")
    group_tpr_curves(df, "results/figures/final_group_tpr.png")

    order = list(CONDITIONS)
    final = df[df.n_labels == df.n_labels.max()]
    cols = ["balanced_accuracy", "fpr", *[f"tpr_{g}" for g in GROUPS], "tpr_gap", "label_quality",
            "labels_spent", "expert_spent", "exposure_max", "exposure_gini"]
    table = final.groupby("variant")[cols].agg(["mean", "std"]).loc[order]
    fmt = pd.DataFrame({c: [f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "—"
                            for m, s in zip(table[(c, "mean")], table[(c, "std")])] for c in cols}, index=order).T
    auc = df.groupby(["variant", "seed"]).apply(
        lambda r: pd.Series({m: area_under_curve(r.n_labels, r[m]) for m in METRICS}), include_groups=False)
    tests = []
    for m in METRICS:
        for a, b in [("final system", "naive HITL"), ("final system", "oracle")]:
            d = auc.xs(a, level="variant")[m] - auc.xs(b, level="variant")[m]
            tests.append({"metric (area under curve)": m, "comparison": f"{a} − {b}",
                          "difference": f"{d.mean():+.3f}", "Wilcoxon p": format_p(wilcoxon(d).pvalue)})
    tests = pd.DataFrame(tests)

    Path("results/tables").mkdir(exist_ok=True)
    with open("results/tables/final.md", "w") as f:
        f.write(f"# Final comparison — 550 labels, mean ± sd over {args.seeds} seeds\n\n")
        f.write(fmt.to_markdown() + "\n\n## Paired tests (area under the learning curve)\n\n")
        f.write(tests.to_markdown(index=False, disable_numparse=True) + "\n")
    print(fmt.to_string())
    print()
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
