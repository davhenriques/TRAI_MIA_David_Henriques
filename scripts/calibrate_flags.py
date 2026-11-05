"""Calibrate the coordinator's flag rule on seeds held apart from evaluation (100-109).

The live rule is tested after every round, so a reviewer counts as flagged if the rule
fires at ANY round (testing 50 times inflates false flags; calibrating on end-of-run
evidence alone underestimates them). In "flags only" mode flags change neither routing nor
labels, so each seed is run once and every candidate rule is replayed on the evidence
available after each round.

Rule: flag reviewer r on group g when P(miss_rg > team_g + margin) > confidence, with the
posterior Beta(m + k * team_g, (n - m) + k * (1 - team_g)), team_g = rest of the team, k = prior.
Criterion: honest flag rate <= 5%; then highest stress-test detection; then earliest.

Usage:  python scripts/calibrate_flags.py
Output: results/runs/flag_calibration.csv
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta

from trai_al.data import GROUPS, load_mhs
from trai_al.experiment import Config, run

PRIORS = [1, 2, 3, 5]
MARGINS = [0.15, 0.2, 0.25]
CONFIDENCES = [0.9, 0.95, 0.99]
RULES = list(itertools.product(PRIORS, MARGINS, CONFIDENCES))
SEEDS = range(100, 110)


def main():
    rows = []
    for seed in SEEDS:
        first_flag = {}  # (rule, reviewer) -> first round the rule fires

        def on_round(round_, learner, c):
            for rid in c.reviewers:
                for g in GROUPS:
                    m, n = c.miss[rid][g]
                    team = c._team_miss(g, without=rid)
                    for rule in RULES:
                        if (rule, rid) in first_flag:
                            continue
                        k, margin, conf = rule
                        p = 1 - beta.cdf(min(team + margin, 1), m + k * team, (n - m) + k * (1 - team))
                        if p > conf:
                            first_flag[(rule, rid)] = round_

        _, _, c, _ = run(Config(labeling="expert+posterior, flags only", seed=seed), load_mhs(seed),
                         return_agents=True, on_round=on_round)
        for rule in RULES:
            for rid in c.reviewers:
                rows.append({"prior": rule[0], "margin": rule[1], "confidence": rule[2], "seed": seed,
                             "reviewer": rid, "flagged": (rule, rid) in first_flag,
                             "round": first_flag.get((rule, rid), np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv("results/runs/flag_calibration.csv", index=False)
    stress = df[df.reviewer == "stress test"].groupby(["prior", "margin", "confidence"])
    honest = df[df.reviewer != "stress test"].groupby(["prior", "margin", "confidence"])
    t = pd.DataFrame({"stress_detected": stress.flagged.mean(), "median_round": stress["round"].median(),
                      "honest_flag_rate": honest.flagged.mean()})
    print(t.round(3).to_string())
    ok = t[t.honest_flag_rate <= 0.05].sort_values(["stress_detected", "median_round"], ascending=[False, True])
    print("\nBest rule with honest flag rate <= 5%:")
    print(ok.head(5).round(3).to_string() if len(ok) else "  none")


if __name__ == "__main__":
    main()
