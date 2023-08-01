import numpy as np
import pandas as pd
from typing import Union
from .antijoin import antijoin


def merge_samples(
    left: pd.DataFrame,
    right: pd.DataFrame,
    dhid: str = "bhid",
    ifrom: str = "from",
    ito: str = "to",
    all_left: bool = True,
) -> pd.DataFrame:
    """ Merge samples from right to left by 'from' and 'to' depths.

    Based on: https://stackoverflow.com/a/44601120/4516267

    Two sample tables (containing downhole depths to the start and end
    of the sample) are combined. The `left` table maintains its records
    and structure and gains the columns of the `right` table (except for
    the 'dhid', 'from', and 'to'). Matching is done on 'dhid' and sample
    depth. If the midpoint of the `left` sample is between the start and
    end of a sample on the `right` it gets the attributes of the `right`
    sample.

    Large samples on the `right` table are 'split' across smaller
    samples on the `left` table.

    Parameters
    ----------
    left : sample table to be updated.
    right : sample table with new attributes (columns).
    dhid : drillhole ID column, either a string common to `left` and
        `right` or a tuple with the `left` and then `right` names as
        strings.
    ifrom : Downhole depth to start of sample, either a string common to
        `left` and `right` or a tuple with the `left` and then `right`
        names as strings.
    ito : Downhole depth to end of sample, either a string common to
        `left` and `right` or a tuple with the `left` and then `right`
        names as strings.
    all_left : keep all records from `left` even if no match in `right`.

    Return
    ------
    `left` data with addition of `right` data's attribute columns. If
    `left` is a pygeostat DataFile a DataFile is returned otherwise a
    Pandas DataFrame.

    """

    # Check the parameters.
    if not (isinstance(left, pd.DataFrame)):
        raise Exception("`left` must be a DataFrame of DataFile.")
    if not (isinstance(right, pd.DataFrame)):
        raise Exception("`right` must be a DataFrame of DataFile.")
    if not (isinstance(dhid, str) or isinstance(dhid, tuple)):
        raise Exception("`dhid` must be a string of tuple of two strings.")
    if not (isinstance(ifrom, str) or isinstance(ifrom, tuple)):
        raise Exception("`ifrom` must be a string of tuple of two strings.")
    if not (isinstance(ito, str) or isinstance(ito, tuple)):
        raise Exception("`ito` must be a string of tuple of two strings.")
    if not isinstance(all_left, bool):
        raise Exception("`all_left` must be `True` or `False`.")

    # Handle left and right column name parameters. Convert all to
    # two-item tuple.
    if not isinstance(dhid, tuple):
        dhid = (dhid, dhid)
    if not isinstance(ifrom, tuple):
        ifrom = (ifrom, ifrom)
    if not isinstance(ito, tuple):
        ito = (ito, ito)

    # Check that the columns are in the DataFrames.
    if not dhid[0] in left.columns:
        raise Exception(f"`dhid`: {dhid[0]} must be in `left`")
    if not dhid[1] in left.columns:
        raise Exception(f"`dhid`: {dhid[1]} must be in `right`")
    if not ifrom[0] in left.columns:
        raise Exception(f"`ifrom`: {ifrom[0]} must be in `left`")
    if not ifrom[1] in left.columns:
        raise Exception(f"`ifrom`: {ifrom[1]} must be in `right`")
    if not ito[0] in left.columns:
        raise Exception(f"`ito`: {ito[0]} must be in `left`")
    if not ito[1] in left.columns:
        raise Exception(f"`ito`: {ito[1]} must be in `right`")

    # Get lists of column names and dictionary of final column names and
    # types.
    # Find columns in `right` that will be added to `left`.
    diff_cols = list(np.setdiff1d(right.columns, left.columns))
    new_cols_dict = right[diff_cols].dtypes.to_dict()
    all_cols = list(left.columns) + diff_cols
    # Add new empty columns with same type as in `right`.
    new_left = left.copy()
    for n, t in new_cols_dict.items():
        new_left[n] = pd.Series(dtype=t)
    all_cols_dict = new_left.dtypes.to_dict()

    # Get arrays from the DataFrames.
    left_mid = (left[ifrom[0]] + ((left[ito[0]] - left[ifrom[0]]) / 2)).values
    right_from = right[ifrom[1]].values
    right_to = right[ito[1]].values
    left_id = left[dhid[0]].values
    right_id = right[dhid[1]].values

    # Remove structure columns from `right` to prevent duplication.
    # Duplication of columns is still possible if there are matching
    # column names. **Maybe test for and remove duplicate column names
    # from the `right` to prevent this**.
    right_drop = right.drop(columns=[dhid[1], ifrom[1], ito[1]])

    # Get indices for `left` and `right` where there is a match. A match
    # is that the `dhid`s match and the midpoint of the `left` sample is
    # between the 'from' and 'to' of the `right` sample.
    i, j = np.where(
        (left_mid[:, None] >= right_from)
        & (left_mid[:, None] < right_to)
        & (left_id[:, None] == right_id)
    )

    # Create merged DataFrame. Need to set type back to original type as
    # this process sets everything to `object` type.
    msamp = pd.DataFrame(
        np.column_stack([left.values[i], right_drop.values[j]]),
        columns=left.columns.append(right_drop.columns),
    ).astype(all_cols_dict)
    # Set specific columns order for append.
    msamp = msamp.loc[:, all_cols]

    # Add unmatched `left` records if requested.
    # Add new empty columns from `right` to `left` to prevent issues
    # with the append when df1.columns != df2.columns.
    if all_left:
        # Get unmatched records from `left` and set specific column
        # order to prevent issues with append.
        unmatched = antijoin(new_left, msamp, on=[dhid[0], ifrom[0], ito[0]])
        unmatched = unmatched.loc[:, all_cols]
        # Append the records.
        msamp = pd.concat(
            [msamp, unmatched],
            ignore_index=True,
            axis=0,
            sort=False,
        )

        # Clean up the index and sort.
        msamp.reset_index(drop=True)
        msamp.sort_values(by=[dhid[0], ifrom[0], ito[0]], inplace=True)

        # Convert back to DataFile, if `left` was.
        if isinstance(left, DataFile):
            msamp = DataFile(data=msamp, dh=dhid, ifrom=ifrom, ito=ito)

    return msamp