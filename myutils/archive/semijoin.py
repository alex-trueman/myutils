import pandas as pd


def semijoin(df1: pd.DataFrame, df2: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Filtering join: extract rows in `df1` with a match in `df2`.

    Alex M Trueman, 2019-06-20

    Return `df1` rows that have a match in `df2` for the given list of
    columns (`cols`). `df1` is filtered by `df2`.
    """
    return df1[df1[cols].apply(tuple, 1).isin(df2[cols].apply(tuple, 1))]
