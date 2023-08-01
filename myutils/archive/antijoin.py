import pandas as pd
from typing import Union

def antijoin(
    left:pd.DataFrame,
    right: pd.DataFrame,
    on: list = None,
    left_on: list = None,
    right_on: list = None) -> pd.DataFrame:
    """Filtering join: extract rows in 'left' with no match in 'right'.

    The `left` frame is filtered by the `right` frame.

    Parameters
    ----------
    left : Data to be filtered.
    right : Data to do the filtering.
    on : List of columns common to `left` and `right` to be used for filtering.
        If this is anything but `None` the parameters `left_on` and `right_on`
        will be ignored.
    left_on : List of columns from `left` to match on. Must be for the same
        column types and in sequence specified with `right_on`.
    right_on : List of columns from `right` to match on. Must be for the same
        column types and in sequence specified with `left_on`.

    Return
    ------
    The data `left` is returned minus rows that have no match in `right` for
    the given list of columns (`on` or `left_on` + `right_on`).

    Example
    -------

    import pandas.util.testing as tm

    # Set default rows and columns for Pandas testing utilities.
    tm.N, tm.K = 5, 3

    # Make dummy DataFrames
    right = tm.makeDataFrame()\
        .set_index(tm.makeCategoricalIndex(k=5, name="bhid"))\
        .reset_index()

    # Convert the `bhid` column to numeric.
    dtfl, dhdict = alpha2numeric(df)

    print("Original DataFrame:\n", df, "\n")
    print("Updated DataFrame:\n", dtfl, "\n")
    print("Hole id dictionary:\n", dhdict)

    """
    
    # Handle the `on` parameters.
    if on is not None:
        left_on = on
        right_on = on
    elif left_on is not None:
        if right_on is None:
            raise Exception("If specifying `left_on` or `right_on` both must be supplied")
        if len(right_on) != len(left_on):
            raise Eception("The `left_on` and `right_on` list must have the same number of items")

    filt = left[~left[left_on].apply(tuple, 1).isin(right[right_on].apply(tuple, 1))]

    return filt
