import pandas as pd
from typing import Tuple, Union


def alpha2numeric(
    data: pd.DataFrame,
    alpha_col: str = "bhid",
    num_col: str = "id",
    to_front: bool = True,
    drop: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Convert an alphanumeric categorical column to numeric codes.

    Alex M Trueman, 2019-07-10

    Automatically assigns numeric codes replacing alphanumeric codes.
    This is the form expected by GSLIB-style programs, which don't work
    with alphanumeric fields. Useful for hole id column and other
    alphanumeric columns typically encountered in data. The function
    returns a dictionary, which maps the numeric code back to the
    alphanumeric code. This can be used to re-assign the alphanumeric
    code at a later time e.g.,:
        `df.replace({"id": dh_dict})`

    Parameters
    ----------
    data: DataFrame or pygeostat DataFile containing column with alphanumeric values.
    alpha_col: Name of the alphanumeric categorical column in `data`.
    num_col: Name of the numeric column to be created. This will
        overwrite any existing column with same name.
    to_front: If `True`, move the column to be the first in `data`.
    drop: If `True`, drop the alphanumeric column from `data`.

    Returns
    -------
    Tuple with 0: modified DataFrame and 1: dictionary mapping the new
    numeric codes (key) to the original alphanumeric codes (value).

    Example
    -------

    import pandas.util.testing as tm

    # Set default rows and columns for Pandas testing utilities.
    tm.N, tm.K = 5, 3

    # Make a dummy DataFrame
    df = tm.makeDataFrame()\
        .set_index(tm.makeCategoricalIndex(k=5, name="bhid"))\
        .reset_index()

    # Convert the `bhid` column to numeric.
    dh, dh_dict = alpha2numeric(df)

    print("Original DataFrame:\n", df, "\n")
    print("Updated DataFrame:\n", dh, "\n")
    print("Hole id dictionary:\n", dh_dict)

    """

    # Copy the input DataFrame to prevent overwriting by reference.
    df: pd.DataFrame
    df = data.copy()

    # Convert column to numeric.
    df[alpha_col] = pd.Categorical(df[alpha_col])
    df[num_col] = df[alpha_col].cat.codes

    # Create dictionary linking the numeric and alphanumeric fields.
    map_dict: dict = dict(pd.Series(df[alpha_col].values, index=df[num_col]))

    if drop:
        df.drop(columns=[alpha_col], inplace=True)

    if to_front:
        cols: list = list(df)
        # Move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index(num_col)))
        # Use loc to reorder
        df = df.loc[:, cols]

    return df, map_dict
