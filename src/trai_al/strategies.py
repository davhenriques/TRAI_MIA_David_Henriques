"""Query strategies: which unlabeled instance(s) to ask a human about.

A strategy scores every candidate (higher = ask first) and explains each choice
in a short reason string, which the learner attaches to its query (transparency).

Baselines: random, uncertainty. With two classes, least confidence, margin and
entropy are monotone in max(p) and rank candidates identically, so one
uncertainty strategy covers all three.
Ours: group-aware uncertainty — uncertain comments that probably target the group
the learner currently serves worst are asked first.
"""

import numpy as np


class RandomSampling:
    name = "random"

    def scores(self, learner, candidates, rng):
        return rng.random(len(candidates))

    def reason(self, learner, pool_index, score):
        return "random sample"


class UncertaintySampling:
    name = "uncertainty"

    def scores(self, learner, candidates, rng):
        proba = learner.predict_proba(candidates)
        uncertainty = 1 - proba.max(axis=1)
        # tiny random jitter breaks exact ties (duplicate texts) without changing the ranking
        return uncertainty + rng.random(len(candidates)) * 1e-9

    def reason(self, learner, pool_index, score):
        return f"model uncertain (1 - max p = {score:.2f})"


class FairUncertaintySampling(UncertaintySampling):
    """score = uncertainty * (1 + lam * relevance), where relevance = sum over groups of
    (group's deficit weight) * P(comment targets that group)."""
    name = "fair"

    def __init__(self, lam=2.0):
        self.lam = lam

    def relevance(self, learner, candidates):
        return sum(w * learner.group_proba(candidates, g) for g, w in learner.group_deficit.items() if w > 0)

    def scores(self, learner, candidates, rng):
        uncertainty = super().scores(learner, candidates, rng)
        return uncertainty * (1 + self.lam * self.relevance(learner, candidates))

    def reason(self, learner, pool_index, score):
        u = 1 - learner.predict_proba([pool_index]).max()
        # name the group contributing most to this comment's relevance
        contrib = {g: w * learner.group_proba([pool_index], g)[0] for g, w in learner.group_deficit.items()}
        g = max(contrib, key=contrib.get)
        return (f"model uncertain (1 - max p = {u:.2f}); probably targets {g} "
                f"(p = {learner.group_proba([pool_index], g)[0]:.2f}), a group it serves below the best "
                f"(est. TPR {learner.est_tpr[g]:.2f} vs {max(learner.est_tpr.values()):.2f})")


STRATEGIES = {s.name: s for s in [RandomSampling, UncertaintySampling, FairUncertaintySampling]}
