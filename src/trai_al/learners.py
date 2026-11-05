"""Learning agents.

`LearnerAgent` is the course's `ActiveLearner` (query / teach / score) recast as an
agent: it keeps its own state (model, labeled set), decides which comments to ask
humans about through a pluggable query strategy, and attaches a reason to every
query so each request can be audited.

To learn under fairness guidance it also keeps:
  - a target-group predictor (which group a comment probably targets), trained on
    the target groups reviewers report together with each label;
  - a self-estimate of how well it serves each group (cross-validated soft TPR on its
    own labeled data — it never sees the test set), shrunk toward the overall rate and
    smoothed across rounds, from which it derives a deficit weight per group;
  - optional reweighing (Kamiran & Calders 2012): labeled examples are weighted so
    that membership in the worst-served group and the label become independent.

Labels may be soft (the coordinator's probability that a comment is hate). A soft
label q is trained as two weighted copies: hate with weight q, not hate with 1 - q.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from trai_al.data import GROUPS


def default_estimator():
    # C=10 selected in the pilot study (scripts/pilot_model_selection.py)
    return LogisticRegression(C=10, max_iter=5000)


@dataclass
class Query:
    """A request for a label, sent by the learner."""
    pool_index: int
    score: float
    reason: str


class LearnerAgent:
    def __init__(self, X_pool, strategy, estimator=None, reweigh=False, rng=None):
        self.X_pool = X_pool
        self.strategy = strategy
        self.estimator = clone(estimator) if estimator is not None else default_estimator()
        self.reweigh = reweigh
        self.rng = rng if rng is not None else np.random.default_rng()
        self.labels = {}  # pool_index -> label (1/0, or a probability of hate)
        self.groups = {}  # pool_index -> {group: bool}, as reported with the label
        self.group_models = {}
        self.est_tpr = None  # set on the first teach()
        self.n_hate_group = {g: 0 for g in GROUPS}  # evidence behind each estimate

    # --- perception -----------------------------------------------------------
    def unlabeled(self):
        mask = np.ones(len(self.X_pool), dtype=bool)
        mask[list(self.labels)] = False
        return np.flatnonzero(mask)

    def predict_proba(self, pool_indices):
        return self.estimator.predict_proba(self.X_pool[pool_indices])

    def group_proba(self, pool_indices, group):
        """Probability that each comment targets `group`."""
        model = self.group_models[group]
        if isinstance(model, float):  # only one class seen so far: use its frequency
            return np.full(len(pool_indices), model)
        return model.predict_proba(self.X_pool[pool_indices])[:, 1]

    @property
    def worst_group(self):
        return min(self.est_tpr, key=self.est_tpr.get)

    def tpr_interval(self, group):
        """95% interval around the self-estimate; its width reflects how few labeled hate
        comments of this group the estimate rests on (shown in the supervisor's report)."""
        t, n = self.est_tpr[group], self.n_hate_group[group] + self.SHRINK
        half = 1.96 * np.sqrt(t * (1 - t) / n)
        return max(0.0, t - half), min(1.0, t + half)

    def pool_tpr_estimate(self):
        """Model-based estimate on the whole pool (labeled or not): expected share of hate
        comments targeting each group that the model detects, using its own P(hate) and the
        target-group predictor. No labels needed, but only as good as the model's calibration."""
        idx = np.arange(len(self.X_pool))
        p = self.predict_proba(idx)[:, 1]
        detected = (p >= 0.5).astype(float)
        out = {}
        for g in GROUPS:
            w = p * self.group_proba(idx, g)  # P(hate and targets g)
            out[g] = float((w * detected).sum() / w.sum())
        w = p
        out["all"] = float((w * detected).sum() / w.sum())
        return out

    @property
    def group_deficit(self):
        """Weight per group: its estimated TPR shortfall relative to the best-served group
        (sums to 1; the best-served group gets 0)."""
        best = max(self.est_tpr.values())
        d = {g: best - t for g, t in self.est_tpr.items()}
        total = sum(d.values())
        return {g: v / total if total > 0 else 1 / len(d) for g, v in d.items()}

    # --- decisions ------------------------------------------------------------
    def query(self, batch_size):
        """Pick the `batch_size` unlabeled comments the strategy ranks highest."""
        candidates = self.unlabeled()
        scores = self.strategy.scores(self, candidates, self.rng)
        top = np.argsort(-scores)[:batch_size]
        return [Query(int(c), float(scores[i]), self.strategy.reason(self, int(c), scores[i]))
                for i, c in zip(top, candidates[top])]

    # --- learning -------------------------------------------------------------
    def teach(self, pool_indices, labels, groups):
        """Add labels (and the target groups reported with them), then retrain."""
        for i, y, g in zip(pool_indices, labels, groups):
            self.labels[int(i)] = float(y)
            self.groups[int(i)] = g
        idx, q = self._labeled_arrays()
        y = (q >= 0.5).astype(int)  # hard view, for the self-estimate and reweighing cells
        G = pd.DataFrame([self.groups[i] for i in idx])[GROUPS].to_numpy(bool)
        self._estimate_group_tpr(idx, y, G)
        self._fit(idx, q, self._weights(y, G))
        self._fit_group_models(idx, G)

    def predict(self, X):
        return self.estimator.predict(X)

    @property
    def n_labels(self):
        return len(self.labels)

    # --- internals ------------------------------------------------------------
    def _labeled_arrays(self):
        idx = np.fromiter(self.labels, dtype=int)
        return idx, np.fromiter(self.labels.values(), dtype=float)

    def _fit(self, idx, q, weights):
        """Fit on (possibly soft) labels: each example becomes a weighted hate copy and a
        weighted not-hate copy; hard labels reduce to ordinary training."""
        w = np.ones(len(idx)) if weights is None else weights
        X = np.vstack([self.X_pool[idx], self.X_pool[idx]])
        y = np.concatenate([np.ones(len(idx), int), np.zeros(len(idx), int)])
        sw = np.concatenate([w * q, w * (1 - q)])
        keep = sw > 0
        self.estimator.fit(X[keep], y[keep], sample_weight=sw[keep])

    def _weights(self, y, G):
        """Kamiran-Calders reweighing on (in worst-served group, label); None = unweighted."""
        if not self.reweigh or self.est_tpr is None:  # no group estimate yet on the first round
            return None
        member = G[:, GROUPS.index(self.worst_group)]
        w = np.ones(len(y))
        for m in (True, False):
            for label in (0, 1):
                cell = (member == m) & (y == label)
                if cell.any():
                    w[cell] = (member == m).mean() * (y == label).mean() / cell.mean()
        return w

    SHRINK = 5    # pseudo-observations pulling a group's estimate toward the overall rate
    SMOOTH = 0.5  # weight of the previous round's estimate

    def _estimate_group_tpr(self, idx, y, G):
        """Soft TPR per group: mean cross-validated P(hate) over the group's labeled hate
        comments, shrunk toward the overall soft TPR (a group with little evidence is not
        automatically 'worst') and smoothed with the previous round's estimate."""
        n_splits = min(5, np.bincount(y, minlength=2).min())
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
            p = cross_val_predict(clone(self.estimator), self.X_pool[idx], y, cv=cv, method="predict_proba",
                                  params={"sample_weight": self._weights(y, G)} if self.reweigh else None)[:, 1]
        else:
            p = np.full(len(y), 0.5)
        overall = p[y == 1].mean()
        new = {}
        for k, g in enumerate(GROUPS):
            hate_g = (y == 1) & G[:, k]
            self.n_hate_group[g] = int(hate_g.sum())
            new[g] = (p[hate_g].sum() + self.SHRINK * overall) / (hate_g.sum() + self.SHRINK)
        if self.est_tpr is None:
            self.est_tpr = new
        else:
            self.est_tpr = {g: self.SMOOTH * self.est_tpr[g] + (1 - self.SMOOTH) * new[g] for g in GROUPS}

    def _fit_group_models(self, idx, G):
        for k, g in enumerate(GROUPS):
            if G[:, k].all() or not G[:, k].any():
                self.group_models[g] = float(G[:, k].mean())
            else:
                self.group_models[g] = default_estimator().fit(self.X_pool[idx], G[:, k])
