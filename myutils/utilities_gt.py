"""Various utility functions for processing and generating grade-tonnage data."""

__author__ = "Alex Trueman"

import numpy as np
import pandas as pd


def sim_gt_stats(
    sim_gt_list,
    tonnes="tonnes",
    grade="grade",
    metal="metal",
    cutoff="cog",
    stat=lambda x: x.quantile(axis=1, q=0.5),
):
    """Calculate statistics across simulation grade-tonnage data.

    These are the statistics of the grade-tonnage-metal at cut-off they
    are not statistics of the realizations themselves.

    Parameters
    ----------
    sim_gt_list : List of DataFrames containing grade-tonnage data for
        a number of simulation realizations. The DataFrames must have a
        cut-off grade column and would normally have columns for tonnage,
        grade, and metal for one or more variables. The dataframes must
        all have identical column names and cut-off grade values.
    tonnes : Name of the tonnes column in the realization
        grade-tonnage data.
    grade : Name of the grade column in the realization
        grade-tonnage data.
    metal : Name of the metal column in the realization
        grade-tonnage data.
    cutoff : Name of the cut-off grade column in the realization
        grade-tonnage data.
    var_dict : Dict of column names present in each of the realization
        grade-tonnage DataFrames. The statistic will be calculated for
        each of the columns. For example:
            `dict(tonnes="Mt", grade="au_ppm", metal="au_Moz")`
    stat : A function applied to the grade-tonnage data with associated
        parameters.

    Returns
    -------
    Dataframe with statistic per variable for each cut-off grade.
    """

    var_in = [tonnes, grade, metal]
    var_out = ["tonnes", "grade", "metal"]
    return (
        pd.DataFrame(
            {
                var: stat(
                    pd.concat(
                        [
                            gt[[cutoff, var]]
                            .rename(columns={var: f"r_{i}"})
                            .set_index(cutoff)
                            for i, gt in enumerate(sim_gt_list)
                        ],
                        axis=1,
                    )
                )
                for var in var_in
            },
        )
        .rename(columns={i: o for i, o in zip(var_in, var_out)})
        .reset_index(drop=False)
    )


def sim_gt_error(
    sim_gt_list,
    tonnes="tonnes",
    grade="grade",
    metal="metal",
    cutoff="cog",
    lower_upper=(0.05, 0.95),
):
    """Calculate simulated error range and median grade-tonnage curves."""
    return {
        k: sim_gt_stats(
            sim_gt_list,
            tonnes=tonnes,
            grade=grade,
            metal=metal,
            cutoff=cutoff,
            stat=lambda x: x.quantile(axis=1, q=q),
        )
        for k, q in zip(["median", "lower", "upper"], (0.5, *lower_upper))
    }


def plot_sim_gt_error(
    axes,
    data_dict,
    c="k",
    ls="-.",
    lw=2.0,
    alpha=0.3,
    zorder=0,
):
    """Plot simulation grade-tonnage ranges on four-subplot layout."""

    # Plot the simulation range as a fill.
    fmt = dict(color=c, ec="none", alpha=alpha, label="Simulation Error", zorder=zorder)
    axes[0].fill_between(
        data_dict["median"]["cog"],
        data_dict["lower"]["tonnes"],
        data_dict["upper"]["tonnes"],
        **fmt,
    )
    axes[1].fill_between(
        data_dict["median"]["cog"],
        data_dict["lower"]["grade"],
        data_dict["upper"]["grade"],
        **fmt,
    )
    axes[2].fill_between(
        data_dict["median"]["cog"],
        data_dict["lower"]["metal"],
        data_dict["upper"]["metal"],
        **fmt,
    )
    # Tonnes-Grade is a special case as x is different for upper and lower.
    # From: https://stackoverflow.com/a/54256731/4516267
    x = np.append(
        data_dict["lower"]["tonnes"].to_numpy(),
        data_dict["upper"]["tonnes"].to_numpy()[::-1],
    )
    y = np.append(
        data_dict["lower"]["grade"].to_numpy(),
        data_dict["upper"]["grade"].to_numpy()[::-1],
    )
    axes[3].fill(x, y, **fmt)

    # Plot the median outcome as a line.
    fmt = dict(
        data=data_dict["median"],
        c=c,
        lw=lw,
        ls=ls,
        label="Simulation Median",
        zorder=zorder + 1,
    )
    axes[0].plot("cog", "tonnes", **fmt)
    axes[1].plot("cog", "grade", **fmt)
    axes[2].plot("cog", "metal", **fmt)
    axes[3].plot("tonnes", "grade", **fmt)
    return axes


def plot_gt(data, axes, label, c="k", ls="solid", lw=2.0, zorder=0):
    """Plot grade-tonnage data on four-subplot layout."""
    axes[0].plot(
        "cog", "tonnes", data=data, c=c, lw=lw, ls=ls, label=label, zorder=zorder
    )
    axes[1].plot("cog", "grade", data=data, c=c, lw=lw, ls=ls, zorder=zorder)
    axes[2].plot("cog", "metal", data=data, c=c, lw=lw, ls=ls, zorder=zorder)
    axes[3].plot("tonnes", "grade", data=data, c=c, lw=lw, ls=ls, zorder=zorder)
    return axes


def plot_metal_curve(
    ax,
    metal_value,
    tonnages,
    grade_func=lambda m, t: (m / t) * 31.1034768,
    metal_label=None,
):
    """Plot a metal curve on a tonnage-grade scatter plot.

    Parameters
    ----------
    ax : Matplotlib Axes where the curve is to be plotted.
    metal_value : The target metal quantity.
    tonnages : 1d-array-like or list of tonnages for the metal curve.
    grade_func : Function for calulating grades from the metal and tonnage values.
    metal_label : String units for metal label. If `None`, no label plotted.

    Return
    ------
    Matplotlib plt of metal curve with label.
    """
    # Calulate grades of the metal curve given the target metal value and
    # a range of tonnages.
    grades = grade_func(metal_value, tonnages)

    ax.plot(tonnages, grades, c="gray", ls="dashed")
    if metal_label:
        anno = ax.annotate(
            f"{metal_value} {metal_label}",
            (tonnages[0], grades[0]),
            ha="left",
            va="top",
        )
        c = ax.get_facecolor()
        anno.set_bbox(dict(facecolor=c, alpha=0.5, edgecolor=c, linewidth=0))
    return ax
