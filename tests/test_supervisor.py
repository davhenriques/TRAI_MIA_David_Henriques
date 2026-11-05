import numpy as np

from trai_al.supervisor import Supervisor
from tests.test_coordinator import Fixed, make_data, religion_blind, truthful
from trai_al.trust import Coordinator


class StubLearner:
    """Only what the report needs."""
    est_tpr = {g: 0.7 for g in ["race", "religion", "origin", "gender", "sexuality"]}
    n_hate_group = {g: 20 for g in est_tpr}

    def tpr_interval(self, g):
        return 0.6, 0.8


def coordinator_with_flag(rounds):
    data = make_data()
    team = [Fixed("blind", religion_blind)] + [Fixed(f"ok{k}", truthful) for k in range(3)]
    c = Coordinator(team, data, np.random.default_rng(1), routing="random", reference="expert",
                    expert_per_round=10, gold_per_round=0, audit_rate=0.0, exclude=False)
    for start in range(0, 10 * rounds, 10):
        c.label_batch(list(range(start, start + 10)))
    return c


def test_supervisor_defers_on_thin_evidence_and_excludes_when_confirmed():
    c = coordinator_with_flag(rounds=40)
    assert "blind" in c.flags and "blind" not in c.excluded  # the coordinator only proposes
    evidence = c.miss["blind"]["religion"][1]
    Supervisor(min_evidence=evidence + 1).review(1, StubLearner(), c)
    assert "blind" not in c.excluded
    assert any(e["event"] == "supervisor_defer" for e in c.log)
    Supervisor(min_evidence=evidence).review(2, StubLearner(), c)
    assert "blind" in c.excluded
    assert any(e["event"] == "supervisor_exclusion" and "religion" in e["reason"] for e in c.log)


def test_no_focus_without_a_credible_gap():
    c = coordinator_with_flag(rounds=2)
    Supervisor().review(1, StubLearner(), c)  # identical intervals: no credible gap
    assert c.focus_group is None
