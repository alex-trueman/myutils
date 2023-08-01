"""Various functions for checking drillhole data tables.

check_dh_table : Basic validation checks of collar, downhole survey, or interval table.
check_intervals : Basic validation checks specific to interval data table.
check_collar_survey : Check for surveys at the collar location.
check_dhids : Report drillholes not present in all drillhole tables.
compare_sample_tables : Compare two drillhole tables for validation.
fill_gaps : Fill gaps in drillhole interval data.
merge_nearest_interval : Merge right interval table with left by nearest downhole depth.
convert_detection_limit : Convert detection limit values to floats.
"""

__author__ = "Alex Trueman"

from itertools import permutations
from typing import Tuple, Union

import numpy as np
import pandas as pd
import rmsp
from rich.console import Console
from rich.style import Style
from rich.text import Text

# Global styles for rich.
style_title = Style(color="black", underline=True, bold=True)
style_pass = Style(color="#3c763d", bgcolor="#dff0d8")
style_warn = Style(color="#996d3b", bgcolor="#fcf8e3")
style_error = Style(color="#a94442", bgcolor="#f2dede")


def _table_checks(table, keys, nan_cols, report):
    """Private: Checks common to all table types."""

    report.append(f"Records count: {table.shape[0]:,}\n")
    rpt_duplicates = table.duplicated(subset=keys).sum()
    report.append(
        f"Duplicate records: {rpt_duplicates:,}\n",
        style=style_error if rpt_duplicates else style_pass,
    )
    if rpt_duplicates:
        report.append(
            "ERROR: Duplicate data should be investigated and corrected.\n",
            style=style_error,
        )

    # Report records with any missing data in required columns.
    rpt_nan = table[nan_cols].shape[0] - table[nan_cols].dropna().shape[0]
    report.append(
        f"Missing values in required columns: {rpt_nan:,}\n",
        style=style_error if rpt_nan else style_pass,
    )

    return report


def _get_column_lists(table):
    """Private: Get lists of required columns for this data type."""

    if isinstance(table, rmsp.CollarData):
        key_cols = [table.dhid, table.x, table.y, table.z]
        nan_cols = [table.dhid, table.x, table.y, table.z]
    elif isinstance(table, rmsp.SurveyData):
        key_cols = [table.dhid, table.depth]
        nan_cols = [table.dhid, table.depth]
    elif isinstance(table, rmsp.IntervalData):
        key_cols = [table.dhid, table.ifrom, table.ito]
        nan_cols = [table.dhid, table.ifrom, table.ito]
    else:
        raise ValueError(
            "Input table must be rmsp CollarsData, SurveyData, or"
            f" IntervalData, not {type(table)}"
        )

    return key_cols, nan_cols


def _check_dhids_data(collars, surveys, intervals):
    """Private: Generate data for `check_dhids`.

    Parameters
    ----------
    collars : rmsp.CollarsData
    surveys " rmsp.SurveysData
    intervals : Dictionary of rmsp.IntervalData where the keys are string
        labels for the tables, e.g., {'assays': assays, 'density': den}.

    Returns
    -------
    Tuple of two dictionaries 1) count of differences; 2) drillhole IDs
    for the differences.
    """

    # Get unique dhids in each table.
    dhids = {n: set(t[t.dhid]) for n, t in intervals.items()}
    dhids["collars"] = set(collars[collars.dhid])
    dhids["surveys"] = set(surveys[surveys.dhid])

    # Return differences for all ordered combinations of sets.
    data = {
        k: v[0] - v[1]
        for k, v in zip(
            permutations(dhids.keys(), 2),
            permutations(dhids.values(), 2),
        )
    }

    # Remove comparisons between interval tables as these aren't useful.
    data = {
        k: v
        for k, v in data.items()
        if k[0] in ["collars", "surveys"] or k[1] in ["collars", "surveys"]
    }
    counts = {k: len(v) for k, v in data.items()}

    return counts, data


def check_dh_table(table):
    """Basic validation checks of collar, downhole survey, or interval table.

    Parameters
    ----------
    table : Collar, survey, or interval table to check.

    Return
    ------
    None. Side effect is printed report of collar table checks.
    """

    if not isinstance(
        table, (rmsp.CollarData, rmsp.SurveyData, rmsp.IntervalData)
    ):
        raise ValueError(
            "Input table must be CollarData, SurveyData, or IntervalData but"
            f" is {type(table).__name__}"
        )

    # Set up report object.
    report = Text()
    report.append(
        f"{type(table).__name__}: Validation Results\n\n", style=style_title
    )

    # Lists of required columns for checking.
    key_cols, nan_cols = _get_column_lists(table)

    # Generate and print report.
    report = _table_checks(table, key_cols, nan_cols, report)
    console = Console()
    console.print(report)

    return None


def check_intervals(table, zero_cols=None):
    """Basic validation checks specific to interval data table."""

    if not isinstance(table, rmsp.IntervalData):
        raise ValueError(
            f"Input table must be IntervalData but is {type(table).__name__}"
        )

    # Set up report object.
    report = Text()
    report.append(
        f"{type(table).__name__} Table Validation Results\n\n",
        style=style_title,
    )

    # Report records with any zero and any negative values in specified columns.
    if zero_cols:
        # Zero values.
        rpt_zero = (
            table[zero_cols].shape[0]
            - table[~(table[zero_cols] == 0).any(axis=1)].shape[0]
        )
        report.append(
            f"Zero value records: {rpt_zero:,}\n",
            style=style_warn if rpt_zero else style_pass,
        )
        if rpt_zero:
            report.append(
                "WARNING: Zero values may be missing assays.\n",
                style=style_warn,
            )
        # Negative values.
        rpt_negative = (
            table[zero_cols].shape[0]
            - table[~(table[zero_cols] < 0).any(axis=1)].shape[0]
        )
        report.append(
            f"Negative value records: {rpt_negative:,}\n",
            style=style_error if rpt_negative else style_pass,
        )
        if rpt_negative:
            report.append(
                (
                    "WARNING: Negative values may be below detection or missing"
                    " assays.\n"
                ),
                style=style_warn,
            )

    # Check for incorrect from/to, gaps, and overlapping samples.
    table = table.sort_values([table.dhid, table.ifrom, table.ito])
    # Records with to <= from.
    rpt_from_to = table[table[table.ito] <= table[table.ifrom]].shape[0]
    report.append(
        f"Interval ito <= ifrom errors: {rpt_from_to:,}\n",
        style=style_error if rpt_from_to else style_pass,
    )
    if rpt_from_to:
        report.append(
            "ERROR: 'to' depth should always be greater than 'from' depth.\n",
            style=style_error,
        )
    # Records with gaps between intervals.
    rpt_gaps = sum(
        (table[table.dhid] == table[table.dhid].shift(-1, axis=0))
        & (table[table.ito] < table[table.ifrom].shift(-1, axis=0))
    )
    report.append(
        f"Interval gaps: {rpt_gaps:,}\n",
        style=style_warn if rpt_gaps else style_pass,
    )
    # Records where intervals overlap.
    rpt_overlap = sum(
        (table[table.dhid] == table[table.dhid].shift(-1, axis=0))
        & (table[table.ito] > table[table.ifrom].shift(-1, axis=0))
        & ~table.duplicated(
            subset=[table.dhid, table.ifrom, table.ito], keep=False
        )
    )
    report.append(
        f"Interval overlaps: {rpt_overlap:,}\n",
        style=style_error if rpt_overlap else style_pass,
    )
    if rpt_overlap:
        report.append(
            "ERROR: Sample intervals should not overlap adjacent intervals.\n",
            style=style_error,
        )

    # Print report.
    console = Console()
    console.print(report)

    return None


def check_collar_survey(collars, surveys):
    """Check for surveys at the collar location.

    Parameters
    ----------
    collars :`rmsp.CollarsData`.
    surveys : `rmsp.SurveyData`.

    Returns
    -------
    Prints a report of number of holes with no survey at the collar.

    Returns list of holes with no survey at the collar.
    """

    col_dh = set(collars[collars.dhid])
    svy_dh = set(surveys.loc[surveys[surveys.depth] == 0, surveys.dhid])

    # No survey at collar.
    data = col_dh - svy_dh
    count = len(data)

    # Generate report.
    report = Text()
    report.append(
        f"No survey at the collar: {count}",
        style=style_warn if count else style_pass,
    )

    console = Console()
    console.print(report)

    return data


def check_dhids(collars, surveys, intervals):
    """Report drillholes not present in all drillhole tables.

    Parameters
    ----------
    collars : rmsp.CollarsData
    surveys " rmsp.SurveysData
    intervals : Dictionary of rmsp.IntervalData where the keys are string
        labels for the tables, e.g., {'assays': assays, 'density': den}.

    Returns
    -------
    Prints counts of differences between sets.

    Returns a dict of drillhole IDs that are the difference between
    set of dhids.
    """

    # Get the data.
    count, data = _check_dhids_data(collars, surveys, intervals)

    # Set up report title and legend.
    report = Text()
    report.append("Drillhole ID check\n\n", style=style_title)
    report.append(" Pass ", style=style_pass)
    report.append(" Warning ", style=style_warn)
    report.append(" Error \n\n", style=style_error)

    # Generate report.
    for k, v in count.items():
        if v == 0:
            style = style_pass
        elif k[1] in ["collars", "surveys"]:
            style = style_error
        else:
            style = style_warn
        label = f"In '{k[0]}' not '{k[1]}'"
        report.append(f"{label:<30}{v:>5}\n", style=style)

    console = Console()
    console.print(report)

    return data


def compare_samp_tables(
    samp1: pd.DataFrame,
    samp2: pd.DataFrame,
    var: Union[str, Tuple[str, str]] = None,
    dhid: Union[str, Tuple[str, str]] = "dhid",
    ifrom: Union[str, Tuple[str, str]] = "from",
    ito: Union[str, Tuple[str, str]] = "to",
) -> dict:
    """Compare two drillhole tables for validation.

    Some useful comparisons of sample statistics for validations after
    operations on the data such as compositing, desurveying, etc.

    Parameters
    ----------
    samp1 : Sample data for comparison.
    samp2 : Sample data for comparison.
    var : Optional grade column for comparing accumulation and grade. Can be a
        two-item tuple if different in `samp1` and `samp2`. `None` if not
        comparing grade column.
    dhid : Drillhole ID column. Can be a two-item tuple if different in `samp1`
        and `samp2`.
    ifrom : Downhole depth to start of sample. Can be a two-item tuple if
        different in `samp1` and `samp2`.
    ito : Downhole depth to end of sample. Can be a two-item tuple if different
        in `samp1` and `samp2`.

    Return
    ------
    A dictionary of comparison statistics.
    """

    # Handle samp1/samp2 parameters. Convert all to two-item tuple.
    if not isinstance(var, tuple):
        var = (var, var)
    if not isinstance(dhid, tuple):
        dhid = (dhid, dhid)
    if not isinstance(ifrom, tuple):
        ifrom = (ifrom, ifrom)
    if not isinstance(ito, tuple):
        ito = (ito, ito)

    # Get only required columns to prevent removing too many records when
    # removing NaNs.
    samp1 = samp1.loc[
        :, [v for v in [dhid[0], ifrom[0], ito[0], var[0]] if v is not None]
    ]
    samp2 = samp2.loc[
        :, [v for v in [dhid[1], ifrom[1], ito[1], var[1]] if v is not None]
    ]
    # Remove row with an NaN value.
    samp1 = samp1.dropna()
    samp2 = samp2.dropna()

    # Count unique drillholes and non-NaN samples.
    samp1_dh_count = samp1[dhid[0]].unique().shape[0]
    samp2_dh_count = samp2[dhid[1]].unique().shape[0]
    samp1_samp_count = samp1.shape[0]
    samp2_samp_count = samp2.shape[0]

    # Accumulate non-NaN sample length and compare.
    samp1_sum_len = round(sum(samp1[ito[0]] - samp1[ifrom[0]]))
    samp2_sum_len = round(sum(samp2[ito[1]] - samp2[ifrom[1]]))

    # Accumulate 'metal' if `var`.
    samp1_sum_metal = None
    samp2_sum_metal = None
    samp1_mean_grade = None
    samp2_mean_grade = None
    if var[0] is not None:
        samp1_sum_metal = round(
            sum((samp1[ito[0]] - samp1[ifrom[0]]) * samp1[var[0]])
        )
        samp2_sum_metal = round(
            sum((samp2[ito[1]] - samp2[ifrom[1]]) * samp2[var[1]])
        )
        samp1_mean_grade = round(samp1_sum_metal / samp1_sum_len, 1)
        samp2_mean_grade = round(samp2_sum_metal / samp2_sum_len, 1)

    report = dict(
        samp1_drillholes=samp1_dh_count,
        samp2_drillholes=samp2_dh_count,
        samp1_samples=samp1_samp_count,
        samp2_samples=samp2_samp_count,
        samp1_length=samp1_sum_len,
        samp2_length=samp2_sum_len,
        samp1_accum=samp1_sum_metal,
        samp2_accum=samp2_sum_metal,
        samp1_grade=samp1_mean_grade,
        samp2_grade=samp2_mean_grade,
    )
    report = {k: v for k, v in report.items() if v is not None}

    return report


def fill_gaps(data):
    """Fill gaps in drillhole interval data.

    Author
    ------
    Alex M Trueman, 2019-07-01

    Description
    -----------
    Fill sample gaps in an IntervalData by creating new records where
    columns of the input IntervalData, except for `dhid`, `from`, and
    `to`, are set to to `NaN`. Set the `from` to the `to` of
    the up-hole record and the `to` to the `from` of the down-hole
    record. Append the new records, sort, and reset the index.

    Parameters
    ----------
    data : rmsp.IntervalData.

    Return
    ------
    rmsp.IntervalData with same columns in same order. New
    records added to fill gaps between samples in the same drillhole.
    """

    # Get column names and order from `data`.
    int_cols = [data.dhid, data.ifrom, data.ito]
    cols = [c for c in data if c not in int_cols]
    # Get the column data types to fix issue with NaN in integer types.
    types = dict(data.dtypes.items())

    # Create a dataframe of gap intervals.
    # A gap is a record where the 'to' depth is less than the next
    # record's 'from' depth and both records have the same dhid.
    gaps = (
        # Get the next record's from depth.
        data.sort_values(int_cols).assign(
            next_from=lambda x: x[data.ifrom].shift(-1, axis=0)
        )
        # Remove all records that are not followed by a gap.
        .loc[
            lambda x: (x[x.dhid] == x[x.dhid].shift(-1, axis=0))
            & (x[x.ito] < x["next_from"])
        ]
        # Reassign intervals depths to define the gap.
        .assign(
            **{
                data.ifrom: lambda x: x[data.ito],
                data.ito: lambda x: x["next_from"],
            },
        )[int_cols]
    )

    # Add gaps intervals into the original DataFrame.
    gap_fill = (
        pd.concat([data, gaps], sort=True)
        .sort_values(int_cols)
        .reset_index(drop=True)[int_cols + cols]
    )

    # Fix integer columns.
    for col, typ in types.items():
        if typ is np.dtype(int):
            gap_fill = gap_fill.assign(
                **{
                    col: lambda x: np.where(
                        x[col].isna(), rmsp.MISSING_INT, x[col]
                    )
                }
            ).astype({col: int})

    return gap_fill


def merge_nearest_interval(
    left,
    right,
    tolerance=1.5,
):
    """Merge right interval table with left by nearest downhole depth.

    Useful for merging interval tables (tables with dhid, from, and to
    depth) where the from and to depths in the two tables do not match.
    Rather than creating new records in the left table, the nearest
    record from the right table is assigned. This method can cause data
    loss and should be checked.

    Both input IntervalData must have the same column names for dhid,
    ifrom, and ito.

    Arguments
    ---------
    left_int : IntervalData to be updated.
    right : IntervalData to merge.
    tolerance : Numeric tolerance parameter for finding nearest interval.
        Can be thought of as a downhole distance within which a nearest
        interval must be found. This parameter should be tested for data
        loss.

    Return
    ------
    The left IntervalData is returned with the addition of columns from the
    right IntervalData.

    """

    # Prepare IntervalData for merging.
    mid_samp = lambda x: x[x.ifrom] + (x[x.ito] - x[x.ifrom]) / 2
    c_right = (
        right.copy()
        .assign(mid=mid_samp)
        .drop(columns=[right.ifrom, right.ito])
        .sort_values("mid")
    )
    c_left = left.copy().assign(mid=mid_samp).sort_values("mid")

    # Merge by nearest midpoint and dhid.
    return (
        pd.merge_asof(
            c_left,
            c_right,
            on="mid",
            by=c_left.dhid,
            tolerance=tolerance,
            direction="nearest",
        )
        .drop(columns=["mid"])
        .sort_values(by=[c_left.dhid, c_left.ifrom, c_left.ito])
    )


def convert_detection_limit(
    data: pd.Series, ldl: str = "<", udl: str = ">", ldl_factor: float = 0.5
) -> np.ndarray:
    """Convert detection limit values to floats.

    Assay data often contain values that are at or below the lower detection
    limit or at or above the upper detection limit of the analysis method.
    Theses values may be identified in the data with alphanumeric codes,
    e.g., '<0.05' for lower detection limit or '>30' for upper detection
    limit.

    This function extracts the numeric part of these values returning the
    input data with converted values. A factor is applied to the lower
    detection limit values. By convention, the lower detection limit
    values are halved, but this can be changed.

    Parameters
    ----------
    data : Pandas series with values to be converted.
    ldl : Prefix that identifies lower detection limit values in the data.
    udl : Prefix that identifies upper detection limit values in the data.
    ldl_factor : Factor applied to lower detection limit values. Set this
        to 1.0 for no change.

    Returns
    -------
    Array of floats with converted detection limit values.

    TODO: Input data must be a pd.Series so that the `startswith()` and
    TODO: `extract()` methods can be used. Find a way to do this same
    TODO: conversion with numpy, allowing more diverse input types.
    """
    return np.where(
        data.str.startswith(ldl, False),
        data.str.extract("(\d+(?:\.\d+)?)", expand=False).astype(float)
        * ldl_factor,
        np.where(
            data.str.startswith(udl, False),
            data.str.extract("(\d+(?:\.\d+)?)", expand=False).astype(float),
            data,
        ),
    ).astype(float)


def convert_detection_limit(
    data: pd.Series, ldl: str = "<", udl: str = ">", ldl_factor: float = 0.5
) -> np.ndarray:
    """Convert detection limit values to floats.

    Assay data often contain values that are at or below the lower detection
    limit or at or above the upper detection limit of the analysis method.
    Theses values may be identified in the data with alphanumeric codes,
    e.g., '<0.05' for lower detection limit or '>30' for upper detection
    limit.

    This function extracts the numeric part of these values returning the
    input data with converted values. A factor is applied to the lower
    detection limit values. By convention, the lower detection limit
    values are halved, but this can be changed.

    Parameters
    ----------
    data : Pandas series with values to be converted.
    ldl : Prefix that identifies lower detection limit values in the data.
    udl : Prefix that identifies upper detection limit values in the data.
    ldl_factor : Factor applied to lower detection limit values. Set this
        to 1.0 for no change.

    Returns
    -------
    Array of floats with converted detection limit values.

    TODO: Input data must be a pd.Series so that the `startswith()` and
    TODO: `extract()` methods can be used. Find a way to do this same
    TODO: conversion with numpy, allowing more diverse input types.
    """
    return np.where(
        data.str.startswith(ldl, False),
        data.str.extract("(\d+(?:\.\d+)?)", expand=False).astype(float)
        * ldl_factor,
        np.where(
            data.str.startswith(udl, False),
            data.str.extract("(\d+(?:\.\d+)?)", expand=False).astype(float),
            data,
        ),
    ).astype(float)


def check_comp_by_cat(dh_raw, dh_comp, var, cat, digits=2, detail_stats=False):
    """Checj sample compositing statistics by category."""

    def _diff(_df, var0, var1):
        return ((_df[var0] - _df[var1]) / ((_df[var1] + _df[var0]) / 2)) * 100

    def _calc_stats(_df, dh_type, var=var, cat=cat):
        return (
            _df.assign(
                **{f"length_{dh_type}": lambda x: x[x.ito] - x[x.ifrom]},
                **{
                    f"accum_{dh_type}": (
                        lambda x, t=dh_type, v=var: x[f"length_{t}"] * x[v]
                    )
                },
            )
            .groupby(cat)[[f"length_{dh_type}", f"accum_{dh_type}"]]
            .sum()
            .assign(
                **{
                    f"grade_{dh_type}": (
                        lambda x, t=dh_type: x[f"accum_{t}"] / x[f"length_{t}"]
                    )
                }
            )
        )

    if detail_stats:
        cols = [
            "length_raw",
            "length_comp",
            "length_diff",
            "accum_raw",
            "accum_comp",
            "accum_diff",
            "grade_raw",
            "grade_comp",
            "grade_diff",
        ]
    else:
        cols = ["length_diff", "accum_diff", "grade_diff"]

    return (
        pd.concat(
            [
                dh_raw.pipe(_calc_stats, dh_type="raw"),
                dh_comp.pipe(_calc_stats, dh_type="comp"),
            ],
            axis=1,
        )
        .assign(
            length_diff=lambda x: _diff(x, "length_raw", "length_comp"),
            accum_diff=lambda x: _diff(x, "accum_raw", "accum_comp"),
            grade_diff=lambda x: _diff(x, "grade_raw", "grade_comp"),
        )
        .round(digits)[cols]
    )


def merge_intervaldata(tables, comp_length=1.0):
    """Merge multiple IntervalData with different downhole intervals.

    This function is useful for combining drillhole intervals of assays and
    other data such as logging data where the sampling intervals may not be
    the same in each. It is not well-tested and may result in data loss.

    Parameters
    ----------
    tables : List of rmsp.IntervalData each with the same columns for 'dhid',
        'ifrom', and 'ito', and each with its own set of attribute columns
        such as 'lith', 'alt', 'au_ppm'.
    comp_length : The length of the composite intervals.

    Returns
    -------
    Merged rmsp.IntervalData.
    """

    # Use compositing to normalize the downhole intervals.
    comp_intervals = rmsp.get_runlength_intervals(tables, comp_length)
    # Composite the input IntervalData into a new list.
    composites = [
        rmsp.composite(tab, comp_intervals, keep_lengths=False)[1]
        for tab in tables
    ]
    # Merge the list of composite IntervalData.
    key = [tables[0].dhid, tables[0].ifrom, tables[0].ito]
    return pd.concat(
        [comp.set_index(key) for comp in composites], axis=1, join="inner"
    ).reset_index()
