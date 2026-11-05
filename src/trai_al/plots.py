"""Figures for the paper (matplotlib, print-friendly).

Colors follow a fixed categorical order validated for color-vision deficiency;
an entity keeps its color in every figure (strategies and groups alike).
"""

import matplotlib.pyplot as plt
import numpy as np

from trai_al.data import GROUPS

VARIANT_COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834", "uncertainty + reweigh": "#eda100",
                  "fair λ=2": "#1baf7a", "fair λ=2 + reweigh": "#6250d6"}
GROUP_COLORS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("labels acquired", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)


def mean_ci(df, column, by):
    """Mean and 95% confidence interval across seeds, per label count."""
    g = df.groupby(by)[column]
    mean, sem = g.mean(), g.sem()
    return mean.index.to_numpy(), mean.to_numpy(), 1.96 * sem.to_numpy()


def learning_curves(df, metrics, path):
    """One panel per metric; one line per variant with a 95% CI band."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2))
    for ax, (column, label) in zip(np.atleast_1d(axes), metrics.items()):
        for variant in [v for v in VARIANT_COLORS if v in set(df.variant)]:
            x, m, ci = mean_ci(df[df.variant == variant], column, "n_labels")
            color = VARIANT_COLORS[variant]
            ax.plot(x, m, color=color, linewidth=2, label=variant)
            ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15, linewidth=0)
        _style(ax, label, label)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def group_tpr_curves(df, path):
    """Detection rate per targeted group, one panel per variant."""
    variants = [v for v in VARIANT_COLORS if v in set(df.variant)]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.2), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), variants):
        sub = df[df.variant == variant]
        for g in GROUPS:
            x, m, _ = mean_ci(sub, f"tpr_{g}", "n_labels")
            ax.plot(x, m, color=GROUP_COLORS[g], linewidth=2, label=g)
        _style(ax, f"Hate detected per group — {variant}", "TPR")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tradeoff(points, path, reference=None):
    """Accuracy vs fairness: one point per lambda (area under curve, mean ± 95% CI).
    `points`: DataFrame with columns lam, acc, acc_ci, gap, gap_ci; `reference`: {name: (acc, gap)}."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.errorbar(points.acc, points.gap, xerr=points.acc_ci, yerr=points.gap_ci, color=VARIANT_COLORS["fair λ=2 + reweigh"],
                linewidth=2, marker="o", markersize=8, capsize=0, elinewidth=1)
    for _, r in points.iterrows():
        ax.annotate(f"λ={r.lam:g}", (r.acc, r.gap), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    for name, (acc, gap) in (reference or {}).items():
        ax.plot(acc, gap, marker="s", markersize=8, color=VARIANT_COLORS[name], linestyle="none", label=name)
    _style(ax, "Accuracy vs fairness (fair + reweigh)", "TPR gap, area under curve (lower is fairer)")
    ax.set_xlabel("balanced accuracy, area under curve", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
