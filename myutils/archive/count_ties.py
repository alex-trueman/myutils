import numpy as np
import pandas as pd


def count_ties(col, min_tied=10):
    """Count the tied values for a DataFrame column.

    Alex M. Trueman, 2019-07-17

    Args:
    ----------
    col: Array-like. DataFrame column, Series, np.ndarray. Contains the data
        to be checked for tied values.
    nrep: The minimum number of tied values required before reporting a value
        to report.

    Return:
    ----------
    Pandas DataFrame with list of tied values, their counts, and proportions
    sorted by count.

    """

    # Get the unique values and the count of each unique value.
    uvals, count = np.unique(col, return_counts=True)

    # Only inspect values where there are >= `min_tied` tied values.
    uvals = uvals[count >= min_tied]
    count = count[count >= min_tied]

    # Get the indexes to sort from highest to lowest count.
    sorder = count.argsort()[::-1]

    # Generate a dataframe of the sorted data.
    c_df = pd.DataFrame({"Value": uvals[sorder], "Count": count[sorder]})
    c_df["Proportion"] = round(c_df["Count"] / len(col), 4)

    return c_df
