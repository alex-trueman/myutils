"""Various utility functions for clustering work."""

__author__ = "Alex Trueman"

from numpy.random import default_rng
import matplotlib.pyplot as plt
from adjustText import adjust_text

def cluster_means(
    data,
    domain,
    var,
    var_labels,
    var_highlight,
    codes,
    labels,
    colors,
    seed=42,
    figsize=(12, 8),
):
    """Plot to compare relative cluster means by feature."""

    rng = default_rng(seed)

    # Vertical width of jitter.
    width = 0.4

    # Get mean by domain for each variable.
    means = data.groupby(domain, as_index=False)[var].agg("mean")

    # List to hold pot annotations.
    texts = []

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for v, l in zip(var, var_labels):
        dots = ax.scatter(
            means[v],
            # Add random jitter to reduce overlap.
            means[domain] + (rng.random(means.shape[0]) * width - width / 2.0),
            marker="o",
            s=100,
            # Map color to domain.
            c=means[domain].map({d: c for d, c in zip(codes, colors)}),
            edgecolors="k",
            linewidths=0.25,
            alpha=1.0 if v in var_highlight else 0.3,
        )
        # Add to list of text annotations.
        texts += [
            ax.text(
                d[0],
                d[1],
                l,
                fontsize="large",
                fontweight="bold" if v in var_highlight else "normal",
            )
            for d in dots.get_offsets().data
        ]

    # Expand the axes limits slightly.
    xlims = ax.get_xlim()
    xmin = xlims[0] - abs(xlims[0]) * 0.1
    xmax = xlims[1] + abs(xlims[1]) * 0.1
    ax.set_xlim(xmin, xmax)
    ylims = ax.get_ylim()
    ymin = ylims[0] - 0.5
    ymax = ylims[1] + 0.5
    ax.set_ylim(ymin, ymax)

    # Add annotations and reduce overlaps.
    adjust_text(texts, expand_points=(2, 2))
    ax.axvline(0, color="k", lw=2, ls="dashed")
    ax.grid(False)
    for key, spine in ax.spines.items():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set(
        title="Per-Feature Cluster Means Relative to Global Means",
        yticks=(codes),
        yticklabels=(labels),
        xticks=([xmin / 2, 0, xmax / 2]),
        xticklabels=(["Below Average", "Global Average", "Above Average"]),
    )
    return fig, ax