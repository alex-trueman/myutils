import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def bxpltgrp(
    data: pd.DataFrame,
    var: str,
    group: str,
    ax: mpl.axes.Axes = None,
    yscale: str = "linear",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    whis=1.5,
    meanline: bool = True,
) -> mpl.axes.Axes:
    """Grouped box plot with sample counts.

    Wrapper around Pandas boxplot with limited options to customize
    the plot. Main thing introduced is labelling categories with the
    number of samples.

    Alex Trueman, 2019-07-30

    Parameters
    ----------
    data : DataFrame or DataFile with variable and group columns.
    var : Column name for box plot statistics.
    group : Column name or list of column names for grouping and
        x-axis display.
    ax : The matplotlib axes to be used.
    yscale : scale for y-axis ("linear", "log", "symlog", "logit").
    xlabel : x-axis label.
    ylabel : y-axis label.
    title : Plot title.
    whis : As a float, determines the reach of the whiskers to the
        beyond the first and third quartiles. In other words, where
        IQR is the interquartile range (Q3-Q1), the upper whisker
        will extend to last datum less than Q3 + whis*IQR).
        Similarly, the lower whisker will extend to the first datum
        greater than Q1 - whis*IQR. Beyond the whiskers, data are
        considered outliers and are plotted as individual points.
        Set this to an unreasonably high value to force the whiskers
        to show the min and max values. Alternatively, set this to an
        ascending sequence of percentile (e.g., [5, 95]) to set the
        whiskers at specific percentiles of the data. Finally, whis
        can be the string 'range' to force the whiskers to the min
        and max of the data.
    meanline : Show a dotted line for the mean.

    Returns
    -------
    matplotlib Axes object.

    """

    # We can only handle one variable at a time.
    if not isinstance(var, str):
        raise Exception("`var` must be a string")

    # Some other checks.
    if not var in data.columns:
        raise Exception("`var` must be in `data`")
    if not isinstance(group, str) and not isinstance(group, list):
        raise Exception("`group` must be a string or list")
    if isinstance(group, str):
        if not group in data.columns:
            raise Exception("`group` must be in `data`")
    else:
        if not all(elem in data.columns for elem in group):
            raise Exception("all columns in `group` must be in `data`")
    if not isinstance(yscale, str):
        raise Exception(
            "`yscale` must be a string and one of 'linear', 'log', 'symlog', or 'logit'"
        )
    if yscale not in ["linear", "log", "symlog", "logit"]:
        raise Exception("`yscale` must be one of 'linear', 'log', 'symlog', or 'logit'")

    # Create the intial boxplot with custom labelling.
    axes = data.boxplot(
        var,
        by=group,
        ax=ax,
        return_type="axes",
        whis=whis,
        meanline=meanline,
        showmeans=meanline,
        flierprops=dict(markerfacecolor="black"),
        medianprops=dict(color="black"),
    )

    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_yscale(yscale)
    plt.suptitle("")

    # Show number of observations in the x-axis tick labels.
    # https://stackoverflow.com/a/29353773/4516267
    if isinstance(group, list):
        keepvars = group + [var]
    else:
        keepvars = [group, var]
    dfg = data[keepvars].groupby(group)
    axes[0].set_xticklabels(["%s\n$n$=%d" % (k, len(v)) for k, v in dfg])

    return axes
