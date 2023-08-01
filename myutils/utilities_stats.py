"""Various statistical analysis functions."""

__author__ = "Alex Trueman"

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def weighted_stats(values, weights) -> Tuple[float, float, float, float]:
    """Return the weighted statistics.

    Modified from: https://stackoverflow.com/a/2415343/4516267
    """

    mean: float = np.average(values, weights=weights)
    variance: float = np.average((values - mean) ** 2, weights=weights)
    std: float = np.sqrt(variance)
    cv: float = std / mean

    return mean, cv, std, variance


def extreme_analysis(
    data,
    var,
    weight,
    log=False,
    title=None,
    figsize=(10, 8),
):
    """Produces various extreme grade analysis plots.

    Produces a set of four subplots for extreme value analysis:
        - The histogram, optionally with log x-axis.
        - The CDF, optionally with log x-axis.
        - Plot of mean grade versus top cut.
        - Plot of coefficient of variation (CV) versus top cut.

    Parameters
    ----------
    data : rmsp GeoDataFrame containing columns `var` and optionally `weight`.
    var : String label of column for variable of interest.
    weight : String label of column for weighting variable.
    log : Boolean, show log x-axis for histogram and CDF.
    title : String, main title above the four plots.
    figsize : Two-tuple, figure size in inches.

    Returns
    -------
    Matplotlib figure and axes objects.


    """

    # Get top cut axis increments from percentiles of `var`.
    topcuts = np.percentile(
        data[var],
        np.concatenate((np.linspace(75, 99, 49), np.linspace(99.1, 99.9, 9))),
    )

    # Weights are optional, set all to 1 if None supplied.
    num_recs = data.shape[0]
    weights = data[weight].values if weight else np.full(num_recs, 1.0)

    # Generate stats for the top cuts.
    tc_mean = np.empty(len(topcuts))
    tc_cv = np.empty(len(topcuts))
    for i, tc in enumerate(topcuts):
        dftc = data[[var]].clip(upper=tc).values.flatten()
        tc_mean[i], _, _, tc_cv[i] = weighted_stats(dftc, weights=weights)

    # Mean and CV of uncut data.
    uc_mean, _, _, uc_cv = weighted_stats(
        data[[var]].values.flatten(), weights=weights
    )

    # Add data point for maximum value to complete the curve.
    max_val = max(data[[var]].values.flatten())
    topcuts = np.append(topcuts, max_val)
    tc_mean = np.append(tc_mean, uc_mean)
    tc_cv = np.append(tc_cv, uc_cv)

    # Make the  plots.
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    data.histplot(
        var=var,
        wt=weight,
        num_bin=30,
        log=log,
        ax=axes[0],
        xlabel="Grade",
        title="Log Histogram" if log else "Histogram",
    )
    data.probplot(
        var=var,
        wt=weight,
        log=log,
        ax=axes[1],
        xlabel="Grade",
        title="Log CDF" if log else "CDF",
        tukey_colors=None,
    )
    axes[2].plot(topcuts, tc_mean, color="k", label="Sensitivity curve")
    axes[3].plot(topcuts, tc_cv, color="k", label="Sensitivity curve")
    axes[2].set_ylabel("Top cut mean")
    axes[2].set_xlabel("Top cut")
    axes[3].set_ylabel("Top cut CV")
    axes[3].set_xlabel("Top cut")
    axes[2].grid(which="both")
    axes[3].grid(which="both")
    axes[2].set_title("Mean Sensitivity")
    axes[3].set_title("CV Sensitivity")
    axes[2].axhline(uc_mean, color="k", linestyle="--", label="Uncut mean")
    axes[3].axhline(uc_cv, color="k", linestyle="--", label="Uncut CV")
    axes[2].legend(loc="lower right")
    axes[3].legend(loc="lower right")
    axes[2].grid(True)
    axes[3].grid(True)

    if title:
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    return fig, axes


def tc_stats(
    df: pd.DataFrame,
    var_orig: str,
    var_tc: str,
    weight: Optional[str] = None,
    group: Optional[Union[str, List[str]]] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """Calculate comparison statistics for top cut data.

    Arguments
    ---------

    df : DataFrame with original and top cut grade columns.
    var_orig : String name of original grade column.
    var_tc : String name of top cut grade column.
    weight : String name of optional weight column for weighting statistics.
    group : String name or list of names of groupby columns for statistics.
    decimals : Rounding decimals.

    Return
    ------

    DataFrame if `group` columns provided otherwise a Series of
    statistics comparing original and top cut variables. Statistics
    reported are:
        n: count of records
        n_tc: count of records affected by top cut
        p_tc: percentage of records affected by top cut
        topcut: top cut value
        min: minimum
        max: maximum
        mean: mean of original (weighted)
        mean_tc: mean of top cut column (weighted)
        cv: coefficient of variation of original (weighted)
        cv_tc: coefficient of variation of top cut column (weighted)

    """

    def f(x: pd.DataFrame, weight: Optional[str] = weight) -> pd.Series:
        """apply function for statistics."""
        d: Dict[str, Union[float, int]] = {"n": x[var_orig].count()}
        d["min"] = x[var_orig].min()
        d["max"] = x[var_orig].max()
        max_tc: float = x[var_tc].max()
        d["topcut"] = max_tc if d["max"] > max_tc else np.NaN
        d["n_tc"] = (x[var_orig] > d["topcut"]).sum()
        d["p_tc"] = d["n_tc"] / d["n"] * 100
        if weight:
            d["mean"], d["cv"], _, _ = weighted_stats(x[var_orig], x[weight])
            d["mean_tc"], d["cv_tc"], _, _ = weighted_stats(
                x[var_tc], x[weight]
            )
        else:
            d["mean"] = x[var_orig].mean()
            d["cv"] = x[var_orig].mean() / x[var_orig].stdev()
            d["mean_tc"] = x[var_tc].mean()
            d["cv_tc"] = x[var_tc].mean() / x[var_tc].stdev()
        return pd.Series(
            d,
            index=[
                "n",
                "min",
                "max",
                "topcut",
                "n_tc",
                "p_tc",
                "mean",
                "cv",
                "mean_tc",
                "cv_tc",
            ],
        )

    # Calculate statistics by group or for entire DataFrame.
    statistics: Union[pd.Series, pd.DataFrame]
    if group:
        statistics = df.groupby(group).apply(f)
    else:
        statistics = df.pipe(f)

    # Clean up data for presentation.
    cols: List[str] = [
        "n",
        "n_tc",
        "p_tc",
        "topcut",
        "min",
        "max",
        "mean",
        "mean_tc",
        "cv",
        "cv_tc",
    ]
    statistics = statistics[cols]
    # Using dictionary for rounding. Possible future option to define
    # rounding per statistic...
    round_cols = {k: decimals for k in cols}
    statistics = statistics.round(round_cols)
    # Change counts to integer so they display with no decimals.
    statistics = statistics.astype({"n": int, "n_tc": int})

    return statistics


def tc_stats_buff(buff, var_orig, var_tc, weight=None, decimals=2):
    """Calculate comparison statistics for top cut data in buffered data.

    Arguments
    ---------
    buff : Dictionary of domain code: DataFrame output from contact modelling.
    var_orig : Name of original grade column.
    var_tc : Name of top cut grade column.
    weight : Name of optional decluster weight column for weighting statistics.
    decimals : Rounding decimals for output data.

    Return
    ------
    DataFrame of statistics comparing original and top cut variables. Statistics
    reported are:
        n: count of records
        n_tc: count of records affected by top cut
        p_tc: percentage of records affected by top cut
        topcut: top cut value
        min: minimum
        max: maximum
        mean: mean of original (weighted)
        mean_tc: mean of top cut column (weighted)
        cv: coefficient of variation of original (weighted)
        cv_tc: coefficient of variation of top cut column (weighted)
    """

    def _f(x, decimals, weight=weight):
        """Function to calculate statistics."""
        d = {"n": x[var_orig].count()}
        d["min"] = x[var_orig].min()
        d["max"] = x[var_orig].max()
        max_tc = x[var_tc].max()
        d["topcut"] = max_tc if d["max"] > max_tc else np.NaN
        d["tc_n"] = (x[var_orig] > d["topcut"]).sum()
        d["tc_%"] = d["tc_n"] / d["n"] * 100
        if weight:
            d["mean"], d["cv"], _, _ = weighted_stats(x[var_orig], x[weight])
            d["mean_tc"], d["cv_tc"], _, _ = weighted_stats(
                x[var_tc], x[weight]
            )
        else:
            d["mean"] = x[var_orig].mean()
            d["cv"] = x[var_orig].mean() / x[var_orig].stdev()
            d["mean_tc"] = x[var_tc].mean()
            d["cv_tc"] = x[var_tc].mean() / x[var_tc].stdev()
        d["mean_%"] = (d["mean_tc"] - d["mean"]) / abs(d["mean"]) * 100
        d["cv_%"] = (d["cv_tc"] - d["cv"]) / abs(d["cv"]) * 100
        return pd.DataFrame({s: [round(v, decimals)] for s, v in d.items()})

    # Calculate statistics for each category in the buffer.
    return pd.concat(
        [
            df[[c for c in [var_orig, var_tc, weight] if c]]
            .dropna()
            .pipe(_f, decimals=decimals, weight=weight)
            .assign(category=cat)
            for cat, df in buff.items()
        ],
        ignore_index=True,
    ).astype({"n": int, "tc_n": int})[
        [
            "category",
            "n",
            "tc_n",
            "tc_%",
            "topcut",
            "min",
            "max",
            "mean",
            "mean_tc",
            "mean_%",
            "cv",
            "cv_tc",
            "cv_%",
        ]
    ]



def weighted_cv(var, weights, mean):
    """Calculated weighted CV."""
    return np.sqrt(np.average((var - mean) ** 2, weights=weights)) / mean


def tc_sensitivity_data(var, weights, topcuts=None):
    """
    Get top cut mean and CV data for sensitivity analysis.

    Parameters
    ----------
    var : 1D array-like grade data.
    weights : 1D array-like declustering weights.
    topcuts : 1D array-like top cut values.

    Returns
    -------
    Tuple of top cuts, mean sensitivity data, and CV sensitivity data.
    """

    # Remove NaNs early.
    var = var[~np.isnan(var)]
    weights = weights[~np.isnan(weights)]

    # Get array of top cut values for testing using percentiles.
    if not topcuts:
        topcuts = list(
            np.percentile(
                var,
                np.concatenate(
                    (np.linspace(75, 99, 49), np.linspace(99.1, 99.9, 9))
                ),
            )
        )

    # Statistics for the top cuts.
    means_tc = [
        np.average(np.clip(var, a_min=None, a_max=tc), weights=weights)
        for tc in topcuts
    ]
    cvs_tc = [
        weighted_cv(np.clip(var, a_min=None, a_max=tc), weights, mn)
        for tc, mn in zip(topcuts, means_tc)
    ]

    # Add data point for maximum value to complete the curve.
    topcuts.append(np.max(var))
    means_tc.append(np.average(var, weights=weights))
    cvs_tc.append(weighted_cv(var, weights, means_tc[-1]))

    return topcuts, means_tc, cvs_tc


def decimated_stats(var, wt, percentile=90):
    """Get mean and CV while decimating the upper tail of the distribution.

    Parameters
    ----------
    var : 1D array-like variables to be decimated.
    wt : 1D array-like weights with same length as `var`.
    percentile : Lowest percentile to decimate to.

    Returns
    -------
    Tuple with np.ndarray means and CVs
    """

    # Remove NaNs early.
    var = var[~np.isnan(var)]
    wt = wt[~np.isnan(wt)]
    if len(var) != len(wt):
        return None

    # Sort the arrays in descending order of var.
    var_idx = var.argsort()
    s_var = var[var_idx[::-1]]
    s_wt = wt[var_idx[::-1]]

    # Find the index of var closest to the chosen lower percentile.
    pcen = np.percentile(s_var, percentile, interpolation="nearest")
    idx_near = abs(s_var - pcen).argmin()

    # Calculate the decimated weighted means and CVs.
    means = [
        np.average(s_var[i:], weights=s_wt[i:]) for i in range(1, idx_near + 1)
    ]
    cvs = [
        weighted_cv(s_var[i:], s_wt[i:], mn)
        for i, mn in zip(range(1, idx_near + 1), means)
    ]

    return means, cvs

def tc_sensitivity_plot(
    var,
    weights,
    topcuts=None,
    stat="mean",
    ax=None,
    figsize=(5, 3),
    y2_label=True,
    log=True,
    legend=False,
):
    """
    Plot mean or CV top cut sensitivity data.

    Parameters
    ----------
    var : 1D array-like grade data.
    weights : 1D array-like declustering weights.
    topcuts : 1D array-like top cut values.
    stat : Statistic to calculate, either 'mean' or 'cv'.
    ax : Matplotlib axes object.
    figsize : Two-tuple, figure size in inches.
    y2_label : Plot the secondary y-axis label?
    log : Log-scale for the x-axis?
    legend : Plot a legend?

    Returns
    -------
    Matplotlib figure and axes objects.
    """

    # Calculate top cut sensitivity statistics.
    topcuts, means, cvs = tc_sensitivity_data(var, weights, topcuts)

    # Make the  plot.
    if not ax:
        _, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
    if stat == "mean":
        # Mean versus TC.
        ax.plot(topcuts, means, color="k", label="Sensitivity curve")
        ax.set_ylabel("Top cut mean")
        ax.set_xlabel("Top cut")
        ax.set_title("Mean Sensitivity")
        ax.axhline(means[-1], color="k", ls="--", label="Uncut mean")
        ax_2 = ax.secondary_yaxis(
            "right",
            functions=(lambda x: x / means[-1], lambda x: x * means[-1]),
        )
        if y2_label:
            ax_2.set_ylabel("Mean normalized to maximum")
        if log:
            ax.set_xscale("log")
    elif stat == "cv":
        # CV versus TC.
        ax.plot(topcuts, cvs, color="k", label="Sensitivity curve")
        ax.set_ylabel("Top cut CV")
        ax.set_xlabel("Top cut")
        ax.set_title("CV Sensitivity")
        ax.axhline(cvs[-1], color="k", ls="--", label="Uncut CV")
        ax_2 = ax.secondary_yaxis(
            "right", functions=(lambda x: x / cvs[-1], lambda x: x * cvs[-1])
        )
        if y2_label:
            ax_2.set_ylabel("CV normalized to maximum")
        if log:
            ax.set_xscale("log")

    if legend:
        ax.legend(loc="lower right")

    return ax

def contingency_table(df, index, columns, wt=None, normalize="index", margins=False):
    """Create a contingency table for two categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame with categorical variables.
    index : Name of column in `df` for the rows of the table.
    columns : Name of column in `df` for the columns of the table.
    wt : Name of column in `df` for weighting of the counts, e.g., decluster weights.
    normalize : See `pandas.crosstab` `normalize` argument.
    margins : See `pandas.crosstab` `margins` argument.

    Return
    ------
    crosstab index, column, and data (as numpy.ndarray).
    """

    x_tab = (
        pd.crosstab(
            index=df[index],
            columns=df[columns],
            values=df[wt],
            aggfunc=sum,
            normalize=normalize,
            margins=margins,
        )
        if wt
        else pd.crosstab(
            index=df[index], columns=df[columns], normalize=normalize, margins=margins
        )
    )

    i_data = x_tab.index
    c_data = x_tab.columns

    return i_data, c_data, x_tab.to_numpy()