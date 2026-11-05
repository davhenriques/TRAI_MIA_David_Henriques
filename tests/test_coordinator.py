import numpy as np
import pandas as pd

from trai_al.data import GROUPS, Split
from trai_al.trust import Coordinator


class Fixed:
    """A reviewer that answers from a fixed rule (no persona needed)."""

    def __init__(self, rid, rule):
        self.rid, self.rule, self.stress_test = rid, rule, False

    def label(self, split, idx):
        return self.rule(split, idx)


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    groups = pd.DataFrame({g: (rng.random(n) < 0.3) if g == "religion" else np.zeros(n, bool) for g in GROUPS})
    split = Split(ids=np.arange(n), X=np.zeros((n, 1)), y=y, groups=groups, score=y * 2.0 - 1,
                  text=np.array([f"comment {i}" for i in range(n)]))

    class D:
        pool = split
        gold = split
    return D


def truthful(split, idx):
    return int(split.y[idx])


def religion_blind(split, idx):
    return 0 if split.groups["religion"].iloc[idx] else int(split.y[idx])


def test_posterior_discounts_a_not_hate_from_a_reviewer_who_misses_that_group():
    data = make_data()
    team = [Fixed("blind", religion_blind)] + [Fixed(f"ok{k}", truthful) for k in range(3)]
    # random routing so the blind reviewer gets enough work to be measured
    c = Coordinator(team, data, np.random.default_rng(0), routing="random", reference="expert",
                    aggregation="posterior", expert_per_round=10, gold_per_round=0, audit_rate=0.0)
    for start in range(0, 300, 10):
        c.label_batch(list(range(start, start + 10)))
    religion_hate = [i for i in range(300, 400) if data.pool.y[i] == 1 and data.pool.groups["religion"].iloc[i]]
    i = religion_hate[0]
    c.answers[i] = {"blind": 0}
    assert c.miss_rate("blind", "religion")[0] > 0.6
    assert c.posterior(i) > 0.3   # a 'not hate' from the blind reviewer is weak evidence
    c.answers[i] = {"ok0": 0}
    assert c.posterior(i) < 0.1   # the same answer from an accurate reviewer is strong evidence


def test_strongly_biased_reviewer_is_flagged_on_the_right_group():
    data = make_data()
    team = [Fixed("blind", religion_blind)] + [Fixed(f"ok{k}", truthful) for k in range(3)]
    c = Coordinator(team, data, np.random.default_rng(1), routing="random", reference="expert",
                    expert_per_round=10, gold_per_round=0, audit_rate=0.0)
    for start in range(0, 400, 10):
        c.label_batch(list(range(start, start + 10)))
    assert c.flags.get("blind", (None,))[0] == "religion"
    assert not any(rid.startswith("ok") for rid in c.flags)


def test_every_query_is_logged():
    data = make_data()
    c = Coordinator([Fixed(f"ok{k}", truthful) for k in range(3)], data, np.random.default_rng(0))
    c.label_batch([0, 1, 2], reasons=["r0", "r1", "r2"])
    logged = [e for e in c.log if e["event"] == "query"]
    assert [e["item"] for e in logged] == [0, 1, 2] and all(e["reason"] for e in logged)


def test_reviewer_is_compared_with_the_rest_of_the_team():
    data = make_data()
    c = Coordinator([Fixed("blind", religion_blind), Fixed("ok", truthful)], data, np.random.default_rng(0))
    c.miss["blind"]["religion"] = [10, 10]
    c.miss["ok"]["religion"] = [0, 10]
    assert c._team_miss("religion", without="blind") < 0.15 < c._team_miss("religion")
