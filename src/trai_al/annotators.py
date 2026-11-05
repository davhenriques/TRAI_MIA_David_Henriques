"""Reviewer agents: AI personas that represent real human annotators.

Each real annotator rated only ~17 comments (~3 outside pool/test), too few to model
one person. Annotators are therefore grouped by behaviour: the corpus' IRT
`annotator_severity`, split into quintiles. In this corpus a high severity means an
annotator flags hate reluctantly (misses more, false-alarms less). Personas are
defined by behaviour only, never by demographics, to avoid encoding stereotypes.

Each persona follows the corpus' own IRT structure:
    P(persona says hate) = sigmoid(a + k * hatefulness + sum_g c_g * targets_g)
  - hatefulness: the comment's latent consensus score. A human reviewer reads the text
    and perceives how hateful it is; the persona models their *response* to it. The
    learner never sees this score.
  - a, k: the persona's threshold and decisiveness;
  - c_g: the persona's extra reluctance (blind spot) for comments targeting group g.
Parameters are fitted on "outside" comments only (never in pool/test), restricted to
clear-consensus comments to match the pool.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trai_al.data import GROUPS

PERSONA_NAMES = ["P1 quick to flag", "P2 fairly quick", "P3 middle", "P4 fairly reluctant", "P5 reluctant to flag"]


def severity_bands(annotations: pd.DataFrame, n_bands: int = 5) -> pd.Series:
    """Persona index (0 = quickest to flag) for every annotator, from severity quintiles."""
    severity = annotations.groupby("annotator_id").annotator_severity.first()
    return pd.Series(pd.qcut(severity, n_bands, labels=False), index=severity.index, name="persona")


def _features(hatefulness, groups):
    return np.column_stack([hatefulness, groups[GROUPS].to_numpy(float)])


class PersonaReviewer:
    """An AI reviewer that labels like the real annotators of one behavioural persona."""

    def __init__(self, name, model, rng):
        self.name = name
        self.model = model
        self.rng = rng

    def proba(self, hatefulness, groups):
        """Probability that this persona would call each comment hate."""
        return self.model.predict_proba(_features(hatefulness, groups))[:, 1]

    def label(self, hatefulness, groups):
        """Sample labels: personas disagree with themselves the way real people do."""
        return (self.rng.random(len(hatefulness)) < self.proba(hatefulness, groups)).astype(int)

    def blind_spots(self):
        """Fitted parameters: threshold, decisiveness and per-group offsets (negative = misses more)."""
        coef = self.model.coef_[0]
        return {"intercept": self.model.intercept_[0], "hatefulness": coef[0],
                **{g: c for g, c in zip(GROUPS, coef[1:])}}


def _fit_persona(data, members=None):
    ann = data.outside_annotations.dropna(subset=["label"])  # 'unclear' answers carry no label
    if members is not None:
        ann = ann[ann.annotator_id.isin(members)]
    rows = pd.Index(data.outside.ids).get_indexer(ann.comment_id)
    # same selection as the pool (clear consensus only): people are far more decisive on
    # clear comments than on ambiguous ones, so fitting on both would overstate hesitation
    clear = data.outside.y[rows] >= 0
    ann, rows = ann[clear], rows[clear]
    groups = data.outside.groups.iloc[rows].reset_index(drop=True)
    X = _features(data.outside.score[rows], groups)
    return LogisticRegression(C=10, max_iter=2000).fit(X, ann.label.to_numpy(int))


def train_personas(data, seed=0, n_bands=5):
    """Train one PersonaReviewer per severity band. Returns (personas, band per annotator)."""
    all_ann = pd.concat([data.outside_annotations, data.annotations])
    bands = severity_bands(all_ann, n_bands)
    rng = np.random.default_rng(seed)
    personas = [PersonaReviewer(PERSONA_NAMES[b], _fit_persona(data, bands.index[bands == b]), rng)
                for b in range(n_bands)]
    return personas, bands


def train_generic_reviewer(data, seed=0):
    """Reference: one reviewer fitted on everyone's labels (no persona structure)."""
    return PersonaReviewer("generic", _fit_persona(data), np.random.default_rng(seed))
