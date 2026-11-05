import numpy as np

from trai_al.data import GROUPS
from trai_al.learners import LearnerAgent
from trai_al.strategies import FairUncertaintySampling, RandomSampling, UncertaintySampling


def groups_for(X):
    """Synthetic target groups: 'religion' when feature 1 is high, nothing else."""
    return [{g: (g == "religion" and x[1] > 0.5) for g in GROUPS} for x in X]


def make_learner(strategy, seed=0, reweigh=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    learner = LearnerAgent(X, strategy, reweigh=reweigh, rng=np.random.default_rng(seed))
    learner.teach(range(40), y[:40], groups_for(X[:40]))
    return learner, y


def test_query_returns_unique_unlabeled_indices_with_reasons():
    learner, _ = make_learner(UncertaintySampling())
    queries = learner.query(10)
    idx = [q.pool_index for q in queries]
    assert len(set(idx)) == 10
    assert not set(idx) & set(learner.labels)
    assert all(q.reason for q in queries)


def test_teach_grows_labeled_set():
    learner, y = make_learner(RandomSampling())
    idx = [q.pool_index for q in learner.query(10)]
    learner.teach(idx, y[idx], groups_for(learner.X_pool[idx]))
    assert learner.n_labels == 50


def test_uncertainty_picks_points_near_the_boundary():
    learner, _ = make_learner(UncertaintySampling())
    q = learner.query(1)[0]
    p = learner.predict_proba([q.pool_index])[0]
    assert abs(p[1] - 0.5) < 0.2


def test_random_is_reproducible():
    a, _ = make_learner(RandomSampling(), seed=3)
    b, _ = make_learner(RandomSampling(), seed=3)
    assert [q.pool_index for q in a.query(5)] == [q.pool_index for q in b.query(5)]


def test_fair_strategy_names_worst_group_in_reason():
    learner, _ = make_learner(FairUncertaintySampling(lam=2), reweigh=True)
    q = learner.query(1)[0]
    assert "probably targets" in q.reason and "est. TPR" in q.reason


def test_reweighing_balances_group_and_label():
    learner, y = make_learner(UncertaintySampling(), reweigh=True)
    learner.est_tpr = {g: 0.1 if g == "religion" else 0.9 for g in GROUPS}  # religion = worst-served
    idx, labels = learner._labeled_arrays()
    G = np.array([[learner.groups[i][g] for g in GROUPS] for i in idx])
    w = learner._weights(labels, G)
    member = G[:, GROUPS.index("religion")]
    assert member.any() and (~member).any()
    # after reweighing, the weighted positive rate is the same inside and outside the group
    rate = lambda m: np.average(labels[m], weights=w[m])
    assert abs(rate(member) - rate(~member)) < 1e-9
