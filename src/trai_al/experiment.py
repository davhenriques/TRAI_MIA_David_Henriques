"""Experiment runner.

Runs the active learning loop for a given config across many seeds and saves
per-round metrics to results/runs/ as CSV, so every paper figure is reproducible.

Labels come either from a consensus oracle (the corpus' aggregated label) or from
the reviewer team (AI personas + one stress-test reviewer) through the coordinator,
configured by one of the LABELING presets below. Reviewers report the targeted groups
together with each label.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trai_al.annotators import train_personas
from trai_al.data import GROUPS, load_mhs
from trai_al.learners import LearnerAgent
from trai_al.metrics import evaluate, gini, labeled_composition
from trai_al.strategies import STRATEGIES
from trai_al.supervisor import Supervisor, report_estimates
from trai_al.trust import Coordinator, Reviewer, stress_test_persona


# coordinator settings per labeling condition
LABELING = {
    "random": dict(routing="random", exclude=False, reference="peer", aggregation="majority"),
    "trust": dict(routing="trust", exclude=False, reference="peer", aggregation="majority"),
    "trust+exclude": dict(routing="trust", exclude=True, reference="peer", aggregation="majority"),
    # v2: unbiased references from an expert (the supervisor) adjudicating a few comments per round
    "expert": dict(routing="trust", exclude=True, reference="expert", aggregation="majority"),
    "expert+posterior": dict(routing="trust", exclude=True, reference="expert", aggregation="posterior"),
    # flags only propose exclusion: nobody acts on them unless a supervisor is present
    "expert+posterior, flags only": dict(routing="trust", exclude=False, reference="expert", aggregation="posterior"),
}


@dataclass
class Config:
    strategy: str = "uncertainty"
    lam: float = 2.0         # fairness weight (fair strategy only)
    reweigh: bool = False    # Kamiran-Calders reweighing when training
    n_initial: int = 50
    budget: int = 500
    batch_size: int = 10
    labeling: str = "oracle"
    stress_group: str = "gender"   # the stress-test reviewer's known blind spot
    stress_offset: float = -3.0    # its strength (P3 middle has about -0.2)
    audit_rate: float = 0.3
    expert_per_round: int = 3      # expert adjudications per round (v2 conditions only)
    supervisor: bool = False       # human on the loop reviewing each round's report
    exposure_cap: float = None     # well-being: max exposure relative to the team's average
    monitor_per_round: int = 0     # expert labels on random held-out pool comments (unbiased monitoring)
    seed: int = 0

    @property
    def variant(self):
        name = f"fair λ={self.lam:g}" if self.strategy == "fair" else self.strategy
        return name + (" + reweigh" if self.reweigh else "")


def initial_sample(y, n, rng):
    """Random initial labeled set; redrawn in the unlikely case it holds a single class."""
    while True:
        idx = rng.choice(len(y), n, replace=False)
        if len(np.unique(y[idx])) == 2:
            return idx


def make_strategy(config):
    cls = STRATEGIES[config.strategy]
    return cls(lam=config.lam) if config.strategy == "fair" else cls()


def make_team(data, config, rng):
    """One reviewer per persona plus one stress-test reviewer with a known blind spot."""
    personas, _ = train_personas(data, seed=config.seed)
    team = [Reviewer(p.name, p) for p in personas]
    stress = stress_test_persona(personas[2], config.stress_group, config.stress_offset)
    stress.rng = rng
    team.append(Reviewer("stress test", stress, stress_test=True))
    for r in team:
        r.persona.rng = rng
    return team


def run(config: Config, data=None, return_agents=False, on_round=None):
    """One active learning run. Returns one row of metrics per round
    (and, with return_agents, the learner and coordinator for inspection).
    `on_round(round, learner, coordinator)` is called after every round (for analyses)."""
    data = data if data is not None else load_mhs(config.seed)
    rng = np.random.default_rng(config.seed)
    learner = LearnerAgent(data.pool.X, make_strategy(config), reweigh=config.reweigh, rng=rng)
    coordinator = None
    if config.labeling != "oracle":
        settings = LABELING[config.labeling]
        coordinator = Coordinator(make_team(data, config, np.random.default_rng(config.seed + 1000)), data,
                                  np.random.default_rng(config.seed + 2000), audit_rate=config.audit_rate,
                                  expert_per_round=config.expert_per_round if settings["reference"] == "expert" else 0,
                                  exposure_cap=config.exposure_cap, monitor_per_round=config.monitor_per_round,
                                  **settings)
    supervisor = Supervisor() if config.supervisor else None

    def get_labels(idx, reasons=None):
        if coordinator is None:  # consensus oracle
            return list(idx), data.pool.y[idx], data.pool.groups.iloc[idx].to_dict("records")
        p_hate = learner.predict_proba(idx)[:, 1] if learner.labels else None
        return coordinator.label_batch(idx, reasons, p_hate)

    init = initial_sample(data.pool.y, config.n_initial, rng)
    learner.teach(*get_labels(init))

    rows = []
    for round_ in range(config.budget // config.batch_size + 1):
        idx = np.fromiter(learner.labels, dtype=int)
        row = {"round": round_, "n_labels": learner.n_labels, "est_worst_group": learner.worst_group,
               **{f"est_tpr_{g}": learner.est_tpr[g] for g in GROUPS},
               **{f"est_lo_{g}": learner.tpr_interval(g)[0] for g in GROUPS},
               **{f"est_hi_{g}": learner.tpr_interval(g)[1] for g in GROUPS},
               **evaluate(learner, data.test), **labeled_composition(learner.labels, data.pool),
               # share of the learner's labels that match the consensus (soft labels read at 0.5),
               # and the mean distance of the (soft) labels from it
               "label_quality": ((np.fromiter(learner.labels.values(), float) >= 0.5) == data.pool.y[idx]).mean(),
               "label_error": np.abs(np.fromiter(learner.labels.values(), float) - data.pool.y[idx]).mean()}
        if coordinator is not None and coordinator.monitor_per_round:
            row.update(report_estimates(learner, coordinator))
        if coordinator is not None:
            stress_flag = coordinator.flags.get("stress test")
            row.update({"labels_spent": coordinator.labels_spent, "expert_spent": coordinator.expert_spent,
                        "n_flagged": len(coordinator.flags),
                        "stress_flagged": stress_flag is not None,
                        "honest_flagged": sum(rid != "stress test" for rid in coordinator.flags),
                        "stress_excluded": "stress test" in coordinator.excluded,
                        "honest_excluded": sum(rid != "stress test" for rid in coordinator.excluded),
                        "focus_group": coordinator.focus_group,
                        "exposure_max": max(coordinator.exposure.values()),
                        "exposure_mean": np.mean(list(coordinator.exposure.values())),
                        "exposure_gini": gini(list(coordinator.exposure.values())),
                        **{f"trust_{rid}": coordinator.trust(rid) for rid in coordinator.reviewers}})
        rows.append(row)
        if on_round is not None:
            on_round(round_, learner, coordinator)
        if round_ == config.budget // config.batch_size:
            break
        queries = learner.query(config.batch_size)
        learner.teach(*get_labels([q.pool_index for q in queries], [q.reason for q in queries]))
        if supervisor is not None:
            changed = supervisor.review(round_ + 1, learner, coordinator)
            if changed[0]:
                learner.teach(*changed)

    df = pd.DataFrame(rows)
    for k, v in asdict(config).items():
        df[k] = v
    df["variant"] = (config.variant if config.labeling == "oracle" else config.labeling
                     + (" + supervisor" if config.supervisor else "")
                     + (f" + cap {config.exposure_cap:g}" if config.exposure_cap else ""))
    return (df, learner, coordinator, supervisor) if return_agents else df


def run_many(configs, seeds) -> pd.DataFrame:
    """Run every config (a dict of Config fields) on every seed."""
    frames = []
    for seed in seeds:
        data = load_mhs(seed)  # same split and initial set for every config within a seed
        for kwargs in configs:
            frames.append(run(Config(seed=seed, **kwargs), data))
    return pd.concat(frames, ignore_index=True)
