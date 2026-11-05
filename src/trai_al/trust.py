"""Coordinator agent: annotator quality control and selection (accountability).

For every comment the learner asks about, the coordinator
  1. routes it to one reviewer (at random, or weighted by trust);
  2. with probability `audit_rate`, asks a second reviewer as a peer check; if they
     disagree, the case is escalated;
  3. each round, slips a few gold items (comments with a highly reliable consensus) to
     reviewers.

Two ways of judging reviewers:
  reference="peer"   (v1) a reviewer is compared with the majority of a peer check. Cheap,
                     but blind to errors the whole team shares: the majority inherits them.
  reference="expert" (v2) a reviewer is only compared with unbiased references: gold items
                     and up to `expert_per_round` comments per round adjudicated by an expert
                     (the human supervisor; simulated by the consensus label). Escalations
                     are adjudicated first.

From those comparisons the coordinator estimates, per reviewer, the miss rate on hate
comments for each targeted group and the false-alarm rate, as Beta posteriors shrunk
toward the team's rates (the evidence behind each estimate stays explicit).

Two ways of turning answers into the learner's label:
  aggregation="majority"   the (peer) majority, as a hard 0/1 label.
  aggregation="posterior"  the probability the comment is hate given who answered what and
                           which group it targets (a group-aware Dawid-Skene step): a "not hate"
                           from a reviewer who misses 45% of religion hate counts for little.

A reviewer is flagged when, with probability > `confidence`, their miss rate on some group
exceeds the team's by more than `bias_margin`. With `exclude=True`, flagged reviewers stop
receiving work and every label their answer decided is re-labeled (remediation); with
`exclude=False` the flag only proposes it and the supervisor decides (`exclude_reviewer`).

Well-being: `est_exposure` is the coordinator's estimate of how much hateful content each
reviewer has read (sum of the learner's P(hate) for pool items; known labels for gold).
With an `exposure_cap` c, a comment the learner thinks is probably hate is not routed to a
reviewer whose estimated exposure exceeds c times the team's average (unless no one else
is left). `exposure` (true hateful comments read) is kept for evaluation only.
The supervisor can set `focus_group`, so expert adjudications prefer comments targeting it.

Monitoring: each round the expert also labels `monitor_per_round` comments drawn at random
from the pool (never queried, never used for training). Unlike the learner's own queries,
this sample is unbiased, so it can check the learner's self-reported performance.
Every event goes to the audit log.
"""

import copy

import numpy as np
from scipy.stats import beta

from trai_al.data import GROUPS


class Reviewer:
    """A member of the reviewer team: an identity plus a labeling policy (persona)."""

    def __init__(self, rid, persona, stress_test=False):
        self.rid = rid
        self.persona = persona
        self.stress_test = stress_test

    def label(self, split, idx):
        return int(self.persona.label(split.score[[idx]], split.groups.iloc[[idx]])[0])


def stress_test_persona(base, group, offset):
    """Synthetic persona: `base` with a strong, known blind spot for `group`."""
    persona = copy.deepcopy(base)
    persona.name = f"stress test ({group} offset {offset:g})"
    persona.model.coef_[0, 1 + GROUPS.index(group)] = offset
    return persona


class Coordinator:
    SHRINK = 5  # pseudo-observations pulling a reviewer's rate toward the team's
    MIN_TEAM = 3  # never exclude below this many active reviewers (peer checks need them)
    FLAG_PRIOR = 1  # pseudo-observations of the rest of the team's rate in the flag test
                    # (prior 1, margin 0.25, confidence 0.95 calibrated on held-out seeds 100-109,
                    # counting flags at any round: scripts/calibrate_flags.py)
    # weak starting belief that reviewers beat chance (~25% misses, ~10% false alarms);
    # real comparisons take over within a few rounds
    MISS_PRIOR = (1, 3)
    ALARM_PRIOR = (1, 9)

    def __init__(self, reviewers, data, rng, routing="trust", exclude=False, audit_rate=0.3,
                 gold_per_round=2, reference="peer", aggregation="majority", expert_per_round=0,
                 bias_margin=0.25, confidence=0.95, trust_power=4, exposure_cap=None, monitor_per_round=0):
        self.reviewers = {r.rid: r for r in reviewers}
        self.data, self.rng = data, rng
        self.routing, self.exclude = routing, exclude
        self.audit_rate, self.gold_per_round = audit_rate, gold_per_round
        self.reference, self.aggregation, self.expert_per_round = reference, aggregation, expert_per_round
        self.bias_margin, self.confidence, self.trust_power = bias_margin, confidence, trust_power
        self.exposure_cap = exposure_cap
        self.exposure = {rid: 0 for rid in self.reviewers}       # true hateful comments read (evaluation only)
        self.est_exposure = {rid: 0.0 for rid in self.reviewers}  # what the coordinator can know
        self.focus_group = None                              # set by the supervisor
        self.monitor_per_round = monitor_per_round
        self.monitor = {}                                    # pool_index -> expert label (held out)
        self._p_hate = 0.0                                   # learner's P(hate) for the item being routed
        self.agree = {rid: [0, 0] for rid in self.reviewers}                       # [agreed, compared]
        self.miss = {rid: {g: [0, 0] for g in GROUPS + ["none"]} for rid in self.reviewers}  # [missed, hate seen]
        self.alarm = {rid: [0, 0] for rid in self.reviewers}                       # [false alarms, not-hate seen]
        self.ref_hate = [0, 0]                                                     # [hate, references]
        self.gold_seen = {rid: set() for rid in self.reviewers}
        self.answers = {}    # pool_index -> {rid: label}
        self.expert = {}     # pool_index -> expert label
        self.flags = {}      # rid -> (group, round)
        self.excluded = set()
        self.labels_spent = 0
        self.expert_spent = 0
        self.log = []
        self.round = 0

    # --- trust and error rates ----------------------------------------------------
    def trust(self, rid):
        agreed, n = self.agree[rid]
        return (1 + agreed) / (2 + n)

    @staticmethod
    def _item_groups(split, idx):
        g = [grp for grp in GROUPS if split.groups[grp].iloc[idx]]
        return g or ["none"]

    def _team_miss(self, group, without=None):
        """Team miss rate on hate targeting `group`; `without` leaves one reviewer out, so a
        reviewer is compared with the rest of the team rather than with a mean they inflate."""
        others = [r for r in self.active() if r != without]
        missed = sum(self.miss[r][group][0] for r in others)
        seen = sum(self.miss[r][group][1] for r in others)
        a, b = self.MISS_PRIOR
        return (a + missed) / (a + b + seen)

    def miss_rate(self, rid, group):
        """Reviewer's miss rate on hate targeting `group`, shrunk toward the team's."""
        missed, n = self.miss[rid][group]
        return (missed + self.SHRINK * self._team_miss(group)) / (n + self.SHRINK), n

    def alarm_rate(self, rid):
        alarms = sum(self.alarm[r][0] for r in self.active())
        seen = sum(self.alarm[r][1] for r in self.active())
        a, b = self.ALARM_PRIOR
        team = (a + alarms) / (a + b + seen)
        a, n = self.alarm[rid]
        return (a + self.SHRINK * team) / (n + self.SHRINK)

    def _record(self, rid, label, reference, split, idx):
        self.agree[rid][0] += int(label == reference)
        self.agree[rid][1] += 1
        if reference == 1:
            for g in self._item_groups(split, idx):
                self.miss[rid][g][0] += int(label == 0)
                self.miss[rid][g][1] += 1
        else:
            self.alarm[rid][0] += int(label == 1)
            self.alarm[rid][1] += 1

    def _check_flags(self):
        """Flag reviewers whose miss rate on a group is credibly far above the team's."""
        new = []
        for g in GROUPS:
            for rid in self.active():
                if rid in self.flags:
                    continue
                team = self._team_miss(g, without=rid)
                m, n = self.miss[rid][g]
                # posterior shrunk toward the rest of the team: a handful of misses is not enough
                a, b = m + self.FLAG_PRIOR * team, (n - m) + self.FLAG_PRIOR * (1 - team)
                p_far_above = 1 - beta.cdf(min(team + self.bias_margin, 1), a, b)
                if p_far_above > self.confidence:
                    self.flags[rid] = (g, self.round)
                    new.append(rid)
                    self._log("flag", reviewer=rid, group=g, reviewer_miss=round((1 + m) / (2 + n), 3),
                              team_miss=round(team, 3), evidence=n, p=round(p_far_above, 3))
        return new

    # --- routing ------------------------------------------------------------------
    def active(self):
        return [rid for rid in self.reviewers if rid not in self.excluded]

    def _route(self, avoid=()):
        candidates = [rid for rid in self.active() if rid not in avoid]
        if not candidates:
            return None
        if self.exposure_cap is not None and self._p_hate > 0.5:
            mean = np.mean([self.est_exposure[r] for r in self.active()])
            rested = [rid for rid in candidates if self.est_exposure[rid] <= self.exposure_cap * mean + 1]
            if rested:
                candidates = rested
        if self.routing == "random":
            return candidates[self.rng.integers(len(candidates))]
        w = np.array([self.trust(rid) ** self.trust_power for rid in candidates])
        return candidates[self.rng.choice(len(candidates), p=w / w.sum())]

    def _ask(self, rid, split, idx, kind):
        label = self.reviewers[rid].label(split, idx)
        self.labels_spent += 1
        self.exposure[rid] += int(split.y[idx] == 1)
        self.est_exposure[rid] += float(split.y[idx]) if split is self.data.gold else self._p_hate
        self._log(kind, reviewer=rid, item=int(idx), label=label, trust=round(self.trust(rid), 3))
        return label

    def _log(self, event, **fields):
        self.log.append({"round": self.round, "event": event, **fields})

    # --- labels for the learner -------------------------------------------------------
    def posterior(self, idx):
        """P(hate | answers, reviewers' group-specific error rates)."""
        if idx in self.expert:
            return float(self.expert[idx])
        prior = (1 + self.ref_hate[0]) / (2 + self.ref_hate[1])
        groups = self._item_groups(self.data.pool, idx)
        like_hate, like_not = prior, 1 - prior
        for rid, y in self.answers[idx].items():
            m = max(self.miss_rate(rid, g)[0] for g in groups)  # the item's most-missed target
            f = self.alarm_rate(rid)
            like_hate *= (1 - m) if y == 1 else m
            like_not *= f if y == 1 else (1 - f)
        return like_hate / (like_hate + like_not)

    def _final(self, idx):
        if idx in self.expert:
            return float(self.expert[idx])
        if self.aggregation == "posterior":
            return self.posterior(idx)
        votes = list(self.answers[idx].values())
        return float(np.mean(votes) > 0.5) if len(votes) != 2 else float(votes[0])  # 2 votes only on agreement

    # --- the coordinator's job -----------------------------------------------------
    def label_batch(self, pool_indices, reasons=None, p_hate=None):
        """Get labels for the learner's queries. Returns (indices, labels, groups),
        including any remediation re-labels the learner must overwrite.
        `p_hate`: the learner's current P(hate) per query (used by the exposure cap)."""
        escalated = []
        for k, idx in enumerate(pool_indices):
            idx = int(idx)
            if reasons is not None:
                self._log("query", item=idx, reason=reasons[k])
            self._p_hate = 0.0 if p_hate is None else float(p_hate[k])
            if self._collect(idx):
                escalated.append(idx)
        self._p_hate = 0.0
        touched = {int(i) for i in pool_indices}
        self._adjudicate(escalated, [int(i) for i in pool_indices])
        self._monitor_sample()
        self._gold_checks()
        for rid in self._check_flags():
            if self.exclude:
                touched |= self._remediate(rid)
        self.round += 1
        idx = sorted(touched)
        return idx, [self._final(i) for i in idx], self.data.pool.groups.iloc[idx].to_dict("records")

    def _collect(self, idx, avoid=()):
        """Ask one reviewer, plus a peer check with probability audit_rate.
        Returns True when the answers disagree (an escalation)."""
        pool = self.data.pool
        first = self._route(avoid)
        answers = {first: self._ask(first, pool, idx, "label")}
        disagree = False
        if self.rng.random() < self.audit_rate and len(self.active()) - len(avoid) > 2:
            second = self._route(avoid=(first, *avoid))
            answers[second] = self._ask(second, pool, idx, "peer_check")
            if answers[second] != answers[first]:
                disagree = True
                self._log("escalation", item=idx, answers=dict(answers))
                if self.reference == "peer" or self.expert_per_round == 0:
                    third = self._route(avoid=(first, second, *avoid))
                    answers[third] = self._ask(third, pool, idx, "tie_break")
            if self.reference == "peer":
                majority = int(np.mean(list(answers.values())) > 0.5)
                for rid, y in answers.items():
                    self._record(rid, y, majority, pool, idx)
        self.answers[idx] = answers
        return disagree

    def _adjudicate(self, escalated, batch):
        """The expert labels escalations first, then a random sample of the batch."""
        if self.reference != "expert" or self.expert_per_round == 0:
            return
        rest = [i for i in batch if i not in escalated and i not in self.expert]
        rest = list(self.rng.permutation(rest))
        if self.focus_group is not None:  # the supervisor asked for audits of this group first
            focus = self.data.pool.groups[self.focus_group].to_numpy()
            rest = [i for i in rest if focus[i]] + [i for i in rest if not focus[i]]
        order = escalated + rest
        pool = self.data.pool
        for idx in order[:self.expert_per_round]:
            truth = int(pool.y[idx])
            self.expert[idx] = truth
            self.expert_spent += 1
            self.ref_hate[0] += truth
            self.ref_hate[1] += 1
            self._log("expert_adjudication", item=int(idx), label=truth, escalated=idx in escalated)
            for rid, y in self.answers[idx].items():
                self._record(rid, y, truth, pool, idx)

    def _monitor_sample(self):
        """Expert labels random, never-queried pool comments: an unbiased held-out sample."""
        free = np.setdiff1d(np.arange(len(self.data.pool)), list(self.answers) + list(self.monitor))
        for idx in self.rng.choice(free, min(self.monitor_per_round, len(free)), replace=False):
            self.monitor[int(idx)] = int(self.data.pool.y[idx])
            self.expert_spent += 1
            self._log("monitor_audit", item=int(idx), label=self.monitor[int(idx)])

    def _gold_checks(self):
        gold = self.data.gold
        for _ in range(self.gold_per_round):
            rid = self.active()[self.rng.integers(len(self.active()))]
            unseen = [i for i in range(len(gold)) if i not in self.gold_seen[rid]]
            if not unseen:
                continue
            i = unseen[self.rng.integers(len(unseen))]
            self.gold_seen[rid].add(i)
            self._record(rid, self._ask(rid, gold, i, "gold"), int(gold.y[i]), gold, i)

    def exclude_reviewer(self, rid):
        """Supervisor's decision: exclude a reviewer now. Returns the pool items whose label
        changed, with their new labels, for the learner to overwrite."""
        affected = sorted(self._remediate(rid))
        return affected, [self._final(i) for i in affected], self.data.pool.groups.iloc[affected].to_dict("records")

    def _remediate(self, rid):
        """Exclude a flagged reviewer; drop their answers and re-ask where they were alone."""
        if len(self.active()) <= self.MIN_TEAM:
            self._log("exclusion_blocked", reviewer=rid, reason=f"team would drop below {self.MIN_TEAM} reviewers")
            return set()
        self.excluded.add(rid)
        affected = {i for i, ans in self.answers.items() if rid in ans and i not in self.expert}
        for i in affected:
            del self.answers[i][rid]
            if not self.answers[i]:
                self._collect(i)
            elif len(self.answers[i]) == 2 and len(set(self.answers[i].values())) == 2:
                tie = self._route(avoid=tuple(self.answers[i]))  # two left, disagreeing: break the tie
                if tie is not None:
                    self.answers[i][tie] = self._ask(tie, self.data.pool, i, "tie_break")
        self._log("exclusion", reviewer=rid, relabeled=len(affected))
        return affected
