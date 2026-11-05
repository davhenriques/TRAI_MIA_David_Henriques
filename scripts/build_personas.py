"""Train the AI reviewer personas and check their fidelity against real annotators.

Fidelity is measured on POOL comments, which no persona saw in training (caveat: the pool comments'
latent score was estimated by the corpus authors from these same annotations, so the
check is somewhat optimistic; persona parameters themselves use outside comments only): for each
persona, its expected miss rate (per targeted group) and false-alarm rate on the same
(comment, annotator) pairs its real members labeled, compared with what those real
members actually answered. A generic reviewer (trained on everyone) is the reference.

Usage:  python scripts/build_personas.py
Output: results/runs/persona_fidelity.csv
"""

import numpy as np
import pandas as pd

from trai_al.annotators import train_generic_reviewer, train_personas
from trai_al.data import GROUPS, load_mhs


def behaviour(p_hate, real, consensus, groups):
    """Miss rate per group and false-alarm rate, real vs expected under the model."""
    hate, not_hate = consensus == 1, consensus == 0
    out = {"real_miss": (real[hate] == 0).mean(), "model_miss": (1 - p_hate[hate]).mean(),
           "real_false_alarm": (real[not_hate] == 1).mean(), "model_false_alarm": p_hate[not_hate].mean()}
    for g in GROUPS:
        m = hate & groups[g].to_numpy()
        out[f"real_miss_{g}"] = (real[m] == 0).mean()
        out[f"model_miss_{g}"] = (1 - p_hate[m]).mean()
    return out


def main():
    data = load_mhs(seed=0)
    personas, bands = train_personas(data)
    generic = train_generic_reviewer(data)

    ann = data.annotations[data.annotations.comment_id.isin(data.pool.ids)].dropna(subset=["label"]).copy()
    ann["persona"] = ann.annotator_id.map(bands)
    rows = pd.Index(data.pool.ids).get_indexer(ann.comment_id)
    hatefulness, consensus = data.pool.score[rows], data.pool.y[rows]
    groups = data.pool.groups.iloc[rows].reset_index(drop=True)
    real = ann.label.to_numpy(int)

    records = []
    for b, persona in enumerate(personas):
        m = (ann.persona == b).to_numpy()
        g = groups[m].reset_index(drop=True)
        for name, reviewer in [("persona", persona), ("generic", generic)]:
            records.append({"persona": persona.name, "model": name, "n_labels": int(m.sum()),
                            **behaviour(reviewer.proba(hatefulness[m], g), real[m], consensus[m], g)})
    df = pd.DataFrame(records)
    df.to_csv("results/runs/persona_fidelity.csv", index=False)

    pd.set_option("display.width", 220)
    p = df[df.model == "persona"].set_index("persona")
    cols = ["miss", "false_alarm"] + [f"miss_{g}" for g in GROUPS]
    table = pd.concat({"real": p[[f"real_{c}" for c in cols]].set_axis(cols, axis=1),
                       "persona": p[[f"model_{c}" for c in cols]].set_axis(cols, axis=1)}, axis=1)
    table = table.swaplevel(axis=1)[cols]
    print("Real annotators vs their persona, on pool comments (miss = said 'not hate' on consensus hate):")
    print(table.round(3).to_string())

    print("\nMean absolute error (real vs model) across personas — persona model vs one generic reviewer:")
    for name in ["persona", "generic"]:
        s = df[df.model == name]
        err = {c: np.mean(np.abs(s[f"real_{c}"] - s[f"model_{c}"])) for c in cols}
        print(f"  {name:8s} " + "  ".join(f"{c}={v:.3f}" for c, v in err.items()))

    print("Persona parameters (hatefulness = decisiveness; group offsets < 0 = extra reluctance / blind spot):")
    print(pd.DataFrame({p.name: p.blind_spots() for p in personas}).T.round(2).to_string())


if __name__ == "__main__":
    main()
