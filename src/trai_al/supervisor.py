"""Supervisor agent: the human on the loop (trust & safety lead).

Each round it reads a report built from what the learner and the coordinator know,
never from the test set:
  - the learner's self-estimated hate detection per targeted group, with a 95% interval
    and the evidence (labeled hate comments) behind it;
  - the reviewer table: trust, flags proposed by the coordinator, exposure;
  - escalations this round.
It also adjudicates escalations as the expert (that part runs inside the coordinator).

Which performance numbers the report shows matters (RQ3): the learner's cross-validation on
its own, actively chosen, labels is badly pessimistic. When the coordinator keeps an
unbiased monitoring sample, the report uses estimator C from `report_estimates` instead
(model-based estimate over the pool, shifted by the bias the monitoring sample reveals).

Oversight policy (a simple, explicit stand-in for a human's judgement):
  1. Exclusions are a human decision. A reviewer flagged by the coordinator is excluded
     only once the evidence is sufficient (>= `min_evidence` hate comments of the flagged
     group compared against unbiased references); until then the decision is deferred.
  2. Fairness-directed audits. If the report shows a credible gap (the worst group's
     interval lies entirely below the best group's), expert adjudications are directed
     to the worst group. The expert budget itself does not change.
Every decision, with its reason, goes to the coordinator's audit log.
"""

import numpy as np
from scipy.stats import beta as beta_dist

from trai_al.data import GROUPS


def report_estimates(learner, coordinator):
    """Three ways to tell the supervisor how well each group is served (RQ3):
    A  monitoring sample: detection on the expert's random held-out comments (Beta interval);
    B  model-based estimate over the whole pool (no labels);
    C  B shifted by A's overall bias (B's precision, A's honesty)."""
    data = coordinator.data
    out = {}
    B = learner.pool_tpr_estimate()
    idx = np.fromiter(coordinator.monitor, dtype=int)
    truth = np.fromiter(coordinator.monitor.values(), dtype=int)
    pred = learner.predict(data.pool.X[idx]) if len(idx) else np.array([], int)
    hate = truth == 1

    def beta_interval(hits, n):
        a, b = 1 + hits, 1 + n - hits
        return a / (a + b), beta_dist.ppf(0.025, a, b), beta_dist.ppf(0.975, a, b)

    all_mean, all_lo, all_hi = beta_interval(pred[hate].sum(), hate.sum())
    shift = all_mean - B["all"]
    for g in GROUPS:
        m = hate & data.pool.groups[g].to_numpy()[idx] if len(idx) else hate
        mean, lo, hi = beta_interval(pred[m].sum(), m.sum())
        c = float(np.clip(B[g] + shift, 0, 1))
        out.update({f"A_{g}": mean, f"A_lo_{g}": lo, f"A_hi_{g}": hi, f"A_n_{g}": int(m.sum()), f"B_{g}": B[g],
                    f"C_{g}": c, f"C_lo_{g}": max(0.0, c - (all_mean - all_lo)),
                    f"C_hi_{g}": min(1.0, c + (all_hi - all_mean))})
    return out


class Supervisor:
    def __init__(self, min_evidence=8):
        self.min_evidence = min_evidence
        self.reports = []
        self._deferred = {}  # rid -> evidence at the last logged deferral

    def report(self, round_, learner, coordinator):
        if coordinator.monitor_per_round and coordinator.monitor:
            est = report_estimates(learner, coordinator)
            tpr = {g: est[f"C_{g}"] for g in GROUPS}
            intervals = {g: (est[f"C_lo_{g}"], est[f"C_hi_{g}"]) for g in GROUPS}
            source = "model estimate over the pool, calibrated on the monitoring sample"
        else:
            tpr = {g: learner.est_tpr[g] for g in GROUPS}
            intervals = {g: learner.tpr_interval(g) for g in GROUPS}
            source = "cross-validation on the learner's own labels (biased: actively chosen hard cases)"
        r = {"round": round_, "source": source,
             "tpr": tpr, "interval": intervals,
             "evidence": dict(learner.n_hate_group),
             "trust": {rid: coordinator.trust(rid) for rid in coordinator.reviewers},
             "pending_flags": {rid: f for rid, f in coordinator.flags.items() if rid not in coordinator.excluded},
             "excluded": sorted(coordinator.excluded),
             "est_exposure": dict(coordinator.est_exposure),
             "escalations": sum(e["event"] == "escalation" and e["round"] == round_ - 1 for e in coordinator.log)}
        self.reports.append(r)
        return r

    def review(self, round_, learner, coordinator):
        """Read the report and act. Returns pool items whose labels changed (for the learner)."""
        r = self.report(round_, learner, coordinator)
        changed = ([], [], [])
        for rid, (group, _) in r["pending_flags"].items():
            evidence = coordinator.miss[rid][group][1]
            if evidence >= self.min_evidence:
                coordinator._log("supervisor_exclusion", reviewer=rid, group=group, evidence=evidence,
                                 reason=f"flag confirmed on {evidence} reference hate comments targeting {group}")
                for part, new in zip(changed, coordinator.exclude_reviewer(rid)):
                    part.extend(new)
            elif self._deferred.get(rid) != evidence:  # log a deferral only when the evidence changed
                self._deferred[rid] = evidence
                coordinator._log("supervisor_defer", reviewer=rid, group=group, evidence=evidence,
                                 reason=f"only {evidence} reference comments; waiting for >= {self.min_evidence}")

        worst = min(r["tpr"], key=r["tpr"].get)
        best = max(r["tpr"], key=r["tpr"].get)
        credible_gap = r["interval"][worst][1] < r["interval"][best][0]
        focus = worst if credible_gap else None
        if focus != coordinator.focus_group:
            coordinator._log("supervisor_focus", group=focus,
                             reason=(f"credible gap: {worst} {r['interval'][worst]} below {best} {r['interval'][best]}"
                                     if focus else "no credible gap: audits back to random"))
            coordinator.focus_group = focus
        return changed
