"""Dataset loading and preprocessing — Measuring Hate Speech corpus (Kennedy et al. 2020).

The corpus has one row per (comment, annotator). This module turns it into:
  - comment-level data for the learner: sentence embeddings + consensus label
    + targeted identity groups, split into pool / test,
  - gold items for the coordinator's quality checks,
  - the real per-annotator labels that reviewer agents replay;
  - "outside" comments (fewer than 3 ratings, never in pool/test) whose real
    annotations train the AI reviewer personas.

Consensus label (from the IRT-based `hate_speech_score`, following the authors' guidance):
  score > 0.5 -> hate (1), score < -1 -> not hate (0), in between -> ambiguous (excluded
  from pool/test, kept separately).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mhs"
CORPUS_URL = ("https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/"
              "resolve/main/measuring-hate-speech.parquet")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HATE_THRESHOLD = 0.5
NOT_HATE_THRESHOLD = -1.0
MIN_RATINGS = 3        # pool/test comments need >= 3 real ratings, so the coordinator can choose reviewers
GOLD_MIN_RATINGS = 50  # the heavily annotated reference comments become gold items

GROUPS = ["race", "religion", "origin", "gender", "sexuality"]

ANNOTATOR_COLUMNS = ["annotator_severity", "annotator_gender", "annotator_educ",
                     "annotator_income", "annotator_ideology", "annotator_age"]


@dataclass
class Split:
    """One set of comments: features, consensus labels and targeted groups."""
    ids: np.ndarray        # comment_id
    X: np.ndarray          # sentence embeddings
    y: np.ndarray          # consensus label: 1 = hate, 0 = not hate
    groups: pd.DataFrame   # bool column per targeted group (majority of annotators)
    score: np.ndarray      # continuous consensus hate_speech_score (IRT)
    text: np.ndarray       # the comment itself (shown to human reviewers in the demo)

    def __len__(self):
        return len(self.ids)


@dataclass
class MHSData:
    pool: Split
    test: Split
    gold: Split
    ambiguous: Split
    annotations: pd.DataFrame          # one row per real annotation of pool/gold/ambiguous comments
    outside: Split                     # comments with < 3 ratings (y = -1 where ambiguous); never pool/test
    outside_annotations: pd.DataFrame  # their real annotations: training data for reviewer personas


def load_corpus() -> pd.DataFrame:
    path = DATA_DIR / "mhs.parquet"
    if not path.exists():
        import urllib.request
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, path)
    return pd.read_parquet(path)


def load_embeddings(comments: pd.DataFrame) -> np.ndarray:
    """Embed each comment once and cache the result (about 30 s on Apple GPU)."""
    path = DATA_DIR / "embeddings_minilm.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        emb = model.encode(comments["text"].tolist(), batch_size=128,
                           normalize_embeddings=True, show_progress_bar=True)
        cached = pd.DataFrame(emb, index=comments.index)
        cached.columns = [f"e{i}" for i in cached.columns]
        cached.to_parquet(path)
    return cached.loc[comments.index].to_numpy(dtype=np.float32)


def build_comments(corpus: pd.DataFrame) -> pd.DataFrame:
    """Aggregate annotations into one row per comment."""
    g = corpus.groupby("comment_id")
    comments = g.agg(text=("text", "first"), score=("hate_speech_score", "first"),
                     std_err=("std_err", "first"), n_ratings=("annotator_id", "size"))
    comments["label"] = np.select(
        [comments.score > HATE_THRESHOLD, comments.score < NOT_HATE_THRESHOLD], [1, 0], default=-1)
    targets = g[[f"target_{grp}" for grp in GROUPS]].mean() > 0.5
    targets.columns = GROUPS
    return comments.join(targets)


def build_annotations(corpus: pd.DataFrame) -> pd.DataFrame:
    """Real per-annotator labels: 1 = hate, 0 = not hate, NaN = annotator chose 'unclear'."""
    ann = corpus[["comment_id", "annotator_id", "hatespeech", *ANNOTATOR_COLUMNS]].copy()
    ann["label"] = ann.pop("hatespeech").map({2.0: 1.0, 0.0: 0.0})
    return ann


def _split(comments: pd.DataFrame, X: np.ndarray, ids) -> Split:
    rows = comments.index.get_indexer(ids)
    sub = comments.iloc[rows]
    return Split(ids=np.asarray(ids), X=X[rows], y=sub["label"].to_numpy(),
                 groups=sub[GROUPS].reset_index(drop=True), score=sub["score"].to_numpy(),
                 text=sub["text"].to_numpy())


def load_mhs(seed: int = 0, test_size: float = 0.25) -> MHSData:
    """Load the corpus and split it. The pool/test split depends on `seed`."""
    corpus = load_corpus()
    comments = build_comments(corpus)
    X = load_embeddings(comments)

    is_gold = comments.n_ratings >= GOLD_MIN_RATINGS
    usable = (comments.n_ratings >= MIN_RATINGS) & ~is_gold
    clear = usable & (comments.label >= 0)

    clear_ids = comments.index[clear]
    pool_ids, test_ids = train_test_split(
        clear_ids, test_size=test_size, stratify=comments.loc[clear_ids, "label"], random_state=seed)
    gold_ids = comments.index[is_gold & (comments.label >= 0)]
    ambiguous_ids = comments.index[usable & (comments.label < 0)]

    outside_ids = comments.index[comments.n_ratings < MIN_RATINGS]

    all_annotations = build_annotations(corpus)
    keep = np.concatenate([pool_ids, gold_ids, ambiguous_ids])
    annotations = all_annotations[all_annotations.comment_id.isin(keep)].reset_index(drop=True)
    outside_annotations = all_annotations[all_annotations.comment_id.isin(outside_ids)].reset_index(drop=True)

    return MHSData(pool=_split(comments, X, pool_ids), test=_split(comments, X, test_ids),
                   gold=_split(comments, X, gold_ids), ambiguous=_split(comments, X, ambiguous_ids),
                   annotations=annotations, outside=_split(comments, X, outside_ids),
                   outside_annotations=outside_annotations)


def describe(data: MHSData) -> pd.DataFrame:
    """Size, hate share and hate comments per targeted group, for each split."""
    rows = {}
    for name in ["pool", "test", "gold", "ambiguous"]:
        s = getattr(data, name)
        hate = s.y == 1
        labeled = name != "ambiguous"  # ambiguous comments have no consensus label
        rows[name] = {"comments": len(s), "hate_share": round(hate.mean(), 3) if labeled else "—",
                      **{f"hate→{grp}": int(s.groups.loc[hate, grp].sum()) if labeled else "—"
                         for grp in GROUPS}}
    return pd.DataFrame(rows).T
