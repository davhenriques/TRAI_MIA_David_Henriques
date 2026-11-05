import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trai_al.annotators import PersonaReviewer, _features
from trai_al.data import GROUPS


def make_persona(seed=0):
    """A persona fitted on synthetic data: flags hate above score 0, less for religion."""
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 2, 2000)
    groups = pd.DataFrame({g: rng.random(2000) < 0.2 for g in GROUPS})
    logit = score - 1.5 * groups["religion"]
    y = (rng.random(2000) < 1 / (1 + np.exp(-logit))).astype(int)
    model = LogisticRegression().fit(_features(score, groups), y)
    return PersonaReviewer("test", model, np.random.default_rng(seed))


def test_more_hateful_comments_are_flagged_more_often():
    p = make_persona()
    none = pd.DataFrame({g: [False, False] for g in GROUPS})
    lo, hi = p.proba(np.array([-2.0, 2.0]), none)
    assert hi > lo


def test_blind_spot_is_recovered():
    assert make_persona().blind_spots()["religion"] < -1


def test_labels_are_reproducible_for_a_seed():
    groups = pd.DataFrame({g: [False] * 50 for g in GROUPS})
    a = make_persona(seed=1).label(np.zeros(50), groups)
    b = make_persona(seed=1).label(np.zeros(50), groups)
    assert (a == b).all()
