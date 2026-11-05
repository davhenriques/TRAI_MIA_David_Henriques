"""RQ3 — how should the system report its own per-group performance to the supervisor?

Compares four estimators against the true per-group TPR on the test set, every round:
  self   cross-validation on the learner's own (actively chosen) labels — the step 6 report
  A      unbiased monitoring sample (expert labels 1 random held-out comment per round)
  B      model-based estimate over the whole pool (no labels)
  C      B shifted by A's overall bias
Setting: coordinator v2 + supervisor, expert budget 2 adjudications + 1 monitoring per round.

Usage:  python scripts/run_report_accuracy.py [--seeds 20]
Output: results/runs/report_accuracy.csv
"""

import argparse

import pandas as pd

from trai_al.data import GROUPS
from trai_al.experiment import run_many


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    df = run_many([{"strategy": "uncertainty", "labeling": "expert+posterior, flags only", "supervisor": True,
                    "expert_per_round": 2, "monitor_per_round": 1}], range(args.seeds))
    df.to_csv("results/runs/report_accuracy.csv", index=False)
    df = df[df["round"] >= 5]  # the monitoring sample needs a few rounds to exist

    rows = []
    for g in GROUPS:
        true = df[f"tpr_{g}"]
        for name, est, lo, hi in [("self", f"est_tpr_{g}", f"est_lo_{g}", f"est_hi_{g}"),
                                  ("A monitoring", f"A_{g}", f"A_lo_{g}", f"A_hi_{g}"),
                                  ("B pool model", f"B_{g}", None, None),
                                  ("C = B + A shift", f"C_{g}", f"C_lo_{g}", f"C_hi_{g}")]:
            r = {"group": g, "estimator": name, "bias": (df[est] - true).mean(), "MAE": (df[est] - true).abs().mean()}
            if lo:
                r["CI_coverage"] = ((df[lo] <= true) & (true <= df[hi])).mean()
                r["CI_width"] = (df[hi] - df[lo]).mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Per group (rounds 5-50, 20 seeds):")
    print(t.round(3).to_string(index=False))
    print("\nAveraged over groups:")
    print(t.groupby("estimator", sort=False)[["bias", "MAE", "CI_coverage", "CI_width"]].mean().round(3).to_string())
    print(f"\nMonitoring hate comments per group at the end (mean): "
          + ", ".join(f"{g} {df[df["round"] == df["round"].max()][f'A_n_{g}'].mean():.1f}" for g in GROUPS))
    print(f"Performance with this budget split: bal-acc at 550 = {df[df.n_labels == 550].balanced_accuracy.mean():.3f}, "
          f"gap = {df[df.n_labels == 550].tpr_gap.mean():.3f}")


if __name__ == "__main__":
    main()
