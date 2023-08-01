"""Various general data handling functions."""

__author__ = "Alex Trueman"

from typing import List

import numpy as np
import pandas as pd
from matplotlib.path import Path


def cols_unique(df: pd.DataFrame, cols: List[str], nans: bool = False) -> np.ndarray:
    """Find unique values across multiple columns (of same type).

    This function looks at the values in all the listed columns and
    creates a sorted array of their unique values.

    Arguments
    ---------
    df : Pandas DataFrame containing columns listed in `cols`.
    cols : List of column names in `df` to compare.
    nans : Return NaNs?

    Return
    ------
    A 1D np.ndarray with unique values across multiple columns.

    """
    u_values: np.ndarray
    # Get 1D array of each column's unique values.
    u_values = np.concatenate(tuple(df[i].unique() for i in cols), axis=0)
    # Remove `NaN`s. Use `pd.isna` as more robust to object (string)
    # type columns with `NaN` values.
    u_values = u_values if nans else u_values[~pd.isna(u_values)]
    # Get the unique values in the array of column unique values.
    # `np.unique` sorts the output.
    u_values = np.unique(u_values)

    return u_values


def polygon_limit(data, x, y, vertices, inside=True, all=None):
    """Select spatial data inside or outside a closed polygon.

    Parameters
    ----------
    data : spatial DataFrame to be restricted to polygon. This object
        normally contains columns for spatial coordinates. The DataFrame
        must have the same number of records as `x` and `y`. If this is
        an rmsp Grid object the coordinates can be derived from:

            x = GridData.coords()[0], for example.

    x, y : 1d array-like coordinates defining locations in a 2d plane.

    vertices : List of tuples. Each tuple contains the 2D coordinate of
        a point on a closed polygon. The first and last tuples should be
        equal. The coordinates should be associated with `data`, `x`,
        and `y`.

    inside : If `True`, data are selected inside the polygon.

    all : Column name string. If defined a Boolean column is returned
        with date defining data inside/outside polygon. If not defined
        only data recods inside/outside the polygon are returned.

    Return
    ------
    Returned object (base type DataFrame) will have same structure and
    meta-data as passed `data` but will be restricted spatially by
    `polygon`.
    """

    polygon = Path(vertices)

    x = x.flatten()
    y = y.flatten()
    points = np.vstack((x, y)).T

    select = (
        polygon.contains_points(points) if inside else ~polygon.contains_points(points)
    )

    if all:
        data[all] = select
    else:
        data = data.loc[select, :].copy()

    return data


def apply_spatial_extents(_df, spatial_extents):
    """Constrain spatial data to spatial extents.

    Parameters
    ----------
    _df : A DataFrame with coordinate columns 'x', 'y', and 'z'.
    spatial_extents : A dictionary of spatial extents with keys of
        coorindate column names (e.g., 'x', 'y', and 'z') and values with
        turples of minimum and maximum coorindates. This type of data
        structure is produced by the rmsp `spatial_extents` method
        (e.g., GridData.spatial_extents).

    Returns
    -------
    `_df` but limited to the spatial extents.

    Example
    -------
    Designed to be used in a `pipe` in DataFrame method chain:

    `grid_small = grid_large.pipe(apply_spatial_extents, mesh_extents)`
    """
    query = " and ".join(
        [
            f"{ax} > {coords[0]} and {ax} < {coords[1]}"
            for ax, coords in spatial_extents.items()
        ]
    )
    return _df.query(query)
