import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rmsp
from typing import Union
from __future__ import annotations


def p_line(
    data: Union[pd.Dataframe, rmsp.GridData, rmsp.SubGridData],
    ax: plt.axes,
    p: Union[float, list[float]] = [99.9, 99.99],
) -> plt.axes:
    """Plot vertical, annotated line at given percentile.

    Alex M. Trueman, 2020-06-15

    Parameters
    ----------
    data : Series or 1D array-like of data.
    ax : Matplotlib axes object where line will be plotted.
    q : List of floats, percentile to calculate and plot.

    Returns
    -------
    Matplotlib axes object.
    """
    if not isinstance(p, list):
        p = [p]

    for pl in p:
        qn = np.nanpercentile(data, q=pl)
        ax.axvline(qn, ls="--")
        ax.annotate(
            f"{pl}%: {round(qn, 1)}",
            xy=(qn, ax.get_ylim()[0]),
            xytext=(2, 10),
            textcoords="offset points",
            va="bottom",
            ha="center",
            rotation=90.0,
            backgroundcolor=ax.get_facecolor(),
        )

    return ax