"""Interactive demo: you join the reviewer team as a real human in the loop.

The system first warms up with its AI reviewers, then puts you on duty. Each round the
learner picks comments it is unsure about (you see why), you label them, the coordinator
cross-checks you against the personas, an expert and hidden gold items, and the
supervisor's report shows how the model serves each targeted group. Everything you do
lands in the audit log.

In `auto` mode a scripted reviewer answers instead of you (for testing, or as a fallback
in a live presentation): "honest" answers like the consensus, "gender-blind" never flags
hate targeting gender — watch the coordinator notice.
"""

import textwrap

import numpy as np
import pandas as pd

from trai_al.data import GROUPS, load_mhs
from trai_al.experiment import Config, initial_sample, make_strategy, make_team
from trai_al.learners import LearnerAgent
from trai_al.supervisor import Supervisor
from trai_al.trust import Coordinator, Reviewer

WARNING = """
CONTENT WARNING — this demo shows real comments from a hate speech corpus, including
slurs and abusive language about race, religion, origin, gender and sexuality.
You can stop at any time by typing q.
"""


class HumanReviewer:
    """A reviewer whose labels come from a person at the keyboard (or a scripted stand-in)."""

    def __init__(self, auto=None, show_gold=False):
        self.rid = "you"
        self.stress_test = False
        self.auto = auto
        self.show_gold = show_gold  # presentation mode: reveal which items are hidden gold checks
        self.coordinator = None  # set once the coordinator exists
        self.learner = None      # set once the learner exists, to explain each request

    def label(self, split, idx):
        is_gold = split is self.coordinator.data.gold
        if self.auto is not None:
            targets_gender = bool(split.groups["gender"].iloc[idx])
            truth = int(split.y[idx])
            return 0 if self.auto == "gender-blind" and targets_gender else truth
        print("\n" + "-" * 78)
        # the same explanation for queries and hidden gold items, so gold checks can't be told apart
        p = self.learner.estimator.predict_proba(split.X[[idx]])[0, 1]
        print(f"why you are asked: the model can't decide (its P(hate) = {p:.2f})")
        if is_gold and self.show_gold:
            print(f"  [GOLD CHECK — hidden from reviewers normally; reference answer: "
                  f"{'hate' if split.y[idx] else 'not hate'}]")
        print(textwrap.fill(str(split.text[idx]), width=78, initial_indent="  ", subsequent_indent="  "))
        while True:
            answer = input("hate speech? [h = hate / n = not hate / q = quit] > ").strip().lower()
            if answer in ("h", "n"):
                return int(answer == "h")
            if answer == "q":
                raise KeyboardInterrupt


class DemoCoordinator(Coordinator):
    """While the human is on duty, they answer first and get one hidden gold item per round."""

    on_duty = False
    _gold_at, _seen_in_batch = None, 0

    def label_batch(self, pool_indices, reasons=None, p_hate=None):
        # on duty, the hidden gold item goes at a random position inside the round, not at the end
        self._gold_at = int(self.rng.integers(len(pool_indices))) if self.on_duty else None
        self._seen_in_batch = 0
        return super().label_batch(pool_indices, reasons, p_hate)

    def _collect(self, idx, avoid=()):
        if self._gold_at is not None and self._seen_in_batch == self._gold_at:
            self._human_gold()
            self._gold_at = None
        self._seen_in_batch += 1
        return super()._collect(idx, avoid)

    def _route(self, avoid=()):
        if self.on_duty and "you" not in avoid and "you" in self.active():
            if self.exposure_cap is None or self._p_hate <= 0.5 or self._under_cap("you"):
                return "you"
            self._log("exposure_cap", reviewer="you", item=None,
                      reason="you have read more hateful content than the team average allows; routed elsewhere")
        return super()._route(avoid=(*avoid, "you"))

    def _under_cap(self, rid):
        mean = np.mean([self.est_exposure[r] for r in self.active()])
        return self.est_exposure[rid] <= self.exposure_cap * mean + 1

    def active(self):
        # off duty, the human is not part of the working team (no routing, no gold checks)
        return [rid for rid in super().active() if self.on_duty or rid != "you"]

    def _gold_checks(self):
        if not self.on_duty:
            return super()._gold_checks()
        # on duty, the human's gold item was already asked inside the batch

    def _human_gold(self):
        gold = self.data.gold
        unseen = [i for i in range(len(gold)) if i not in self.gold_seen["you"]]
        if unseen:
            i = unseen[self.rng.integers(len(unseen))]
            self.gold_seen["you"].add(i)
            self._record("you", self._ask("you", gold, i, "gold"), int(gold.y[i]), gold, i)


def _print_round(round_, coordinator, supervisor, learner, start):
    events = pd.DataFrame(coordinator.log[start:])
    mine = events[events.get("reviewer", pd.Series(dtype=str)) == "you"] if "reviewer" in events else events.iloc[0:0]
    print("\n" + "=" * 78)
    print(f"ROUND {round_} — what happened")
    for _, e in events.iterrows():
        if e.event == "escalation" and "you" in e.answers:
            others = {k: v for k, v in e.answers.items() if k != "you"}
            print(f"  ! a persona disagreed with you on comment {e['item']}: {others} -> escalated to the expert")
        elif e.event == "expert_adjudication" and e["item"] in coordinator.answers \
                and "you" in coordinator.answers[e["item"]]:
            yours = coordinator.answers[e["item"]]["you"]
            verdict = "agreed with you" if yours == e.label else "DISAGREED with you"
            print(f"  expert checked comment {e['item']}: {'hate' if e.label else 'not hate'} ({verdict})")
        elif e.event == "exposure_cap":
            print(f"  (well-being) a likely-hateful comment was routed to someone else: {e.reason}")
        elif e.event in ("flag", "supervisor_exclusion", "supervisor_defer") and e.get("reviewer") == "you":
            print(f"  >> {e.event.replace('_', ' ')}: {e.get('reason') or e.get('group')}")
    gold = mine[mine.event == "gold"]
    for _, e in gold.iterrows():
        truth = coordinator.data.gold.y[e["item"]]
        print(f"  hidden gold item: you said {'hate' if e.label else 'not hate'}, "
              f"reference {'hate' if truth else 'not hate'}")

    agreed, n = coordinator.agree["you"]
    print(f"\nYOUR TRUST: {coordinator.trust('you'):.2f} ({agreed}/{n} answers matched unbiased references; "
          "trust starts at 0.50 and moves with the evidence)")
    rows = []
    for g in GROUPS:
        m, ev = coordinator.miss_rate("you", g)
        rows.append(f"{g} {m:.2f} (n={ev})" if ev else f"{g} no evidence yet")
    print("your estimated miss rate on hate, per targeted group: " + ", ".join(rows))
    print(f"hateful comments you have read: {coordinator.exposure['you']} "
          f"(team average {np.mean(list(coordinator.exposure.values())):.1f})")

    r = supervisor.reports[-1]
    print(f"\nSUPERVISOR REPORT — source: {r['source']}")
    for g in GROUPS:
        lo, hi = r["interval"][g]
        print(f"  {g:10s} hate detected {r['tpr'][g]:.2f}  [{lo:.2f}, {hi:.2f}]")
    if r["pending_flags"]:
        print(f"  pending flags: {r['pending_flags']}")
    print(f"  excluded reviewers: {r['excluded'] or 'none'}")


def run_demo(rounds=3, batch_size=5, warmup_rounds=10, auto=None, seed=0, exposure_cap=None, verbose=True,
             show_gold=False):
    """Run the demo. Returns (learner, coordinator, supervisor) for inspection."""
    say = print if verbose else (lambda *a, **k: None)
    if auto is None:
        print(WARNING)
        input("press ENTER to continue > ")
    data = load_mhs(seed)
    config = Config(labeling="expert+posterior, flags only", supervisor=True, expert_per_round=2,
                    monitor_per_round=1, seed=seed)
    rng = np.random.default_rng(seed)
    human = HumanReviewer(auto, show_gold)
    team = make_team(data, config, np.random.default_rng(seed + 1000)) + [human]
    coordinator = DemoCoordinator(team, data, np.random.default_rng(seed + 2000), routing="trust", exclude=False,
                                  reference="expert", aggregation="posterior", expert_per_round=2,
                                  monitor_per_round=1, exposure_cap=exposure_cap)
    human.coordinator = coordinator
    supervisor = Supervisor()
    learner = LearnerAgent(data.pool.X, make_strategy(config), rng=rng)
    human.learner = learner

    say("warming up: the AI reviewers label the first comments ...")
    init = initial_sample(data.pool.y, config.n_initial, rng)
    learner.teach(*coordinator.label_batch(init))
    for r in range(warmup_rounds):
        q = learner.query(10)
        learner.teach(*coordinator.label_batch([x.pool_index for x in q], [x.reason for x in q],
                                               learner.predict_proba([x.pool_index for x in q])[:, 1]))
        supervisor.review(r + 1, learner, coordinator)
    say(f"warm-up done: {learner.n_labels} labeled comments. You are now on duty.\n")

    coordinator.on_duty = True
    try:
        for r in range(rounds):
            start = len(coordinator.log)
            q = learner.query(batch_size)
            idx = [x.pool_index for x in q]
            learner.teach(*coordinator.label_batch(idx, [x.reason for x in q], learner.predict_proba(idx)[:, 1]))
            changed = supervisor.review(warmup_rounds + r + 1, learner, coordinator)
            if changed[0]:
                learner.teach(*changed)
            if verbose:
                _print_round(r + 1, coordinator, supervisor, learner, start)
    except KeyboardInterrupt:
        print("\nstopped.")
    coordinator.on_duty = False
    say("\nAUDIT LOG — your last entries:")
    log = pd.DataFrame(coordinator.log)
    say(log[log.get("reviewer") == "you"].tail(8)[["round", "event", "item", "label", "trust"]].to_string(index=False))
    return learner, coordinator, supervisor
