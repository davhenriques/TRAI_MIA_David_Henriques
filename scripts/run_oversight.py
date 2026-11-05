"""RQ3 (oversight & transparency) and RQ4 (reviewer well-being).

Conditions (uncertainty sampling, coordinator v2: expert references + corrected labels):
  auto            flagged reviewers excluded automatically (step 5 result)
  no oversight    flags are only proposals and nobody acts on them
  supervisor      the supervisor decides exclusions (evidence threshold) and directs audits
  supervisor+cap  as above, plus an exposure cap (1.2 x team average) on likely-hate comments

Usage:  python scripts/run_oversight.py [--seeds 20]
Output: results/runs/oversight.csv, results/figures/oversight_*.png
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from trai_al.data import GROUPS, load_mhs
from trai_al.experiment import run_many
from trai_al.metrics import area_under_curve
from trai_al.plots import VARIANT_COLORS, learning_curves

FLAGS_ONLY = "expert+posterior, flags only"
CONDITIONS = {
    "auto": dict(labeling="expert+posterior"),
    "no oversight": dict(labeling=FLAGS_ONLY),
    "supervisor": dict(labeling=FLAGS_ONLY, supervisor=True),
    "supervisor+cap": dict(labeling=FLAGS_ONLY, supervisor=True, exposure_cap=1.2),
}
METRICS = ["balanced_accuracy", "fpr", "tpr_gap", "worst_group_tpr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    df = run_many([{"strategy": "uncertainty", **c} for c in CONDITIONS.values()], range(args.seeds))
    names = {f"{c['labeling']}" + (" + supervisor" if c.get("supervisor") else "")
             + (f" + cap {c['exposure_cap']:g}" if c.get("exposure_cap") else ""): k for k, c in CONDITIONS.items()}
    df["variant"] = df.variant.map(names)
    df.to_csv("results/runs/oversight.csv", index=False)
    print(f"{args.seeds} seeds x {len(CONDITIONS)} conditions in {time.time() - t0:.0f}s")

    VARIANT_COLORS.clear()
    VARIANT_COLORS.update(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
    learning_curves(df, {"balanced_accuracy": "Balanced accuracy", "tpr_gap": "TPR gap (lower is fairer)",
                         "exposure_max": "Max hateful comments read by one reviewer"},
                    "results/figures/oversight_curves.png")

    order = list(CONDITIONS)
    auc = df.groupby(["variant", "seed"]).apply(
        lambda r: pd.Series({m: area_under_curve(r.n_labels, r[m]) for m in METRICS}), include_groups=False)
    print("\nRQ3 — area under curve (mean ± sd):")
    print(auc.groupby("variant").agg(["mean", "std"]).round(3).loc[order].to_string())
    for a, b in [("supervisor", "no oversight"), ("supervisor", "auto"), ("supervisor+cap", "supervisor")]:
        d = auc.xs(a, level="variant") - auc.xs(b, level="variant")
        print(f"  {a} minus {b}: " + "  ".join(f"{m} {d[m].mean():+.3f} (p={wilcoxon(d[m]).pvalue:.3f})" for m in METRICS))

    final = df[df.n_labels == df.n_labels.max()]
    print("\nExclusions at the end (mean over seeds): stress-test excluded share, honest reviewers excluded per run")
    print(final.groupby("variant")[["stress_excluded", "honest_excluded", "stress_flagged", "honest_flagged"]]
          .mean().round(2).loc[order].to_string())

    print("\nRQ3 — is the supervisor's report accurate? (self-estimated TPR per group vs true test TPR, all rounds)")
    sup = df[df.variant == "supervisor"]
    rows = []
    for g in GROUPS:
        inside = (sup[f"est_lo_{g}"] <= sup[f"tpr_{g}"]) & (sup[f"tpr_{g}"] <= sup[f"est_hi_{g}"])
        rows.append({"group": g, "mean_est": sup[f"est_tpr_{g}"].mean(), "mean_true": sup[f"tpr_{g}"].mean(),
                     "MAE": (sup[f"est_tpr_{g}"] - sup[f"tpr_{g}"]).abs().mean(), "CI_coverage": inside.mean(),
                     "mean_CI_width": (sup[f"est_hi_{g}"] - sup[f"est_lo_{g}"]).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    true_worst = sup[[f"tpr_{g}" for g in GROUPS]].idxmin(axis=1).str.replace("tpr_", "")
    focused = sup.focus_group.notna()
    print(f"  rounds with a credible-gap focus: {focused.mean():.0%}; focus = true worst group in "
          f"{(sup.focus_group[focused] == true_worst[focused]).mean():.0%} of them")

    print("\nRQ4 — exposure to hateful content at 550 labels (true counts, mean over seeds):")
    print(final.groupby("variant")[["exposure_max", "exposure_mean", "exposure_gini"]].mean().round(2).loc[order].to_string())


if __name__ == "__main__":
    main()
