"""Various functions for processing simulated data."""

__author__ = "Alex Trueman"

import numpy as np
from pandas import DataFrame
from satsp import solver
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from typing import Union, Tuple, List


def sort_realizations(
    data: Union[DataFrame, np.ndarray],
    num_real: int,
    num_x: int,
    num_y: int,
    window_size: int = 1,
    distance_metric: str = "euclidean",
) -> Tuple[List[int], np.ndarray]:
    """Sort simulation realizations by similarity.

    Continuous or categorical simulation realizations for a 2D plane are
    ordered using a distance metric. The ordered realizations will
    produce a smoother transition when the realization images are viewed
    as an animation. Simulated annealing is used to find the shortest
    'distance' between images.

    Arguments
    ---------
    data : Realization data for one variable organized as GSLIB-like grid
        with realizations appended end-on-end (row-wise). The data must
        represent a 2D grid (or a 2D orthogonal slice of a 3D grid).
    num_real : Number of simulation realizations in `data`.
    num_x : Number of grid nodes in x-axis orientation of the 2D grid.
    num_y : Number of grid nodes in y-axis orientation of the 2D grid.
    window : i Number of nodes offset for moving window applied in x and
        y-axes. Default value of 1 is appropriate for categorial variables.
    distance_metric : A valid scipy pdist metric name.

    Returns
    -------
    2-tuple:
        0. Realization indices in an optimized order for animations.
        1. Distance matrix for all pairs of realizations.
    """

    # A median filter is applied to continuous variable simulations to
    # simplify the features of the image (`window_size > 1`). Categorical
    # data shouldn't be filtered (`window_size <= 1`).
    for i in range(num_real):
        if window_size != 1:
            data[i] = ndimage.median_filter(
                np.reshape(data[i], ((num_x, num_y))), window_size
            ).flatten()

    # Calculate distances between all pairs of realizations.
    distance_matrix: np.ndarray = squareform(
        pdist(np.nan_to_num(data), metric=distance_metric)  # type: ignore
    )

    # Calculate optimal ordering of realizations.
    solver.Solve(dist_matrix=distance_matrix, screen_output=False)
    # Solver returns 1-indexed list.
    optimal_order: List[int] = [t - 1 for t in solver.GetBestTour()]

    return optimal_order, distance_matrix


def select_scenarios(gt_data, real="real", cog="cog", metal="metal", cl=(0.1, 0.9)):
    """Scenario reduction based on quantiles of metal at cut-off grades.
    
    The set of simulation realizations is reduced to three: low case, median
    case, and high case. These are selected by finding the realizations that 
    have the closest metal value to the the median and an upper and lower
    quantile at each cut-off grade.
    
    Parameters
    ----------
    gt_data : DataFrame of tonnage-grade-metal values at cut-off grades
        for all realizations. Must contain the columns defined by the
        following arguments.
    real : String label of the column with realization indices.
    cog : String label of column containing the cut-off grade values.
    metal : String label of the column continaing metal values at
        cut-off grade, While metal is the recommended metric here
        any sensible column could be passed (e.g., tonnes, grade, NSR).
    cl : 2-tuple with values of the lower and upper quantiles, in that
        order.
    
    Returns
    -------
    A 3-tuple with the realization index of the closest matching lower, median,
    and upper scenarios, in that order.
    """

    # Get quantiles of metal across realizations at each cut-off grade.
    cog_quantiles = (
        gt_data.groupby(cog)
        .agg(
            low=(metal, lambda x: np.quantile(x, q=cl[0])),
            medium=(metal, lambda x: np.median(x)),
            high=(metal, lambda x: np.quantile(x, q=cl[1])),
        )
        .reset_index(drop=False)
    )

    # Calculate distance between each realization's metal and quantiles
    # at cut-off.
    # TODO: Investigate better metrics for distance. Perhaps absolute distance.
    distance = gt_data.merge(cog_quantiles[[cog, "low", "medium", "high"]], on=cog)
    distance["low_diff"] = distance[metal] - distance["low"]
    distance["median"] = distance[metal] - distance["medium"]
    distance["high_diff"] = distance[metal] - distance["high"]

    # Get the absolute sum of the differences.
    distance = (
        distance.groupby(real)
        .agg(
            low_case_distance=("low_diff", lambda x: abs(sum(x))),
            median_case_distance=("median", lambda x: abs(sum(x))),
            high_case_distance=("high_diff", lambda x: abs(sum(x))),
        )
        .reset_index(drop=False)
    )

    # Get the realizations closest to the quantiles.
    low_case = int(distance.iloc[distance["low_case_distance"].idxmin()]["real"])
    median_case = int(distance.iloc[distance["median_case_distance"].idxmin()]["real"])
    high_case = int(distance.iloc[distance["high_case_distance"].idxmin()]["real"])

    # Return list of realization indices for each quantile.
    return low_case, median_case, high_case