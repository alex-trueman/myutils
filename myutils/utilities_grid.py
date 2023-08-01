"""Various functions for working with grid data."""


import itertools
__author__ = "Alex Trueman"

import numpy as np
from typing import Union, Tuple
import rmsp


def grid_to_meshgrid(
    grid: rmsp.GridData, var: str, default: Union[int, float], plane: str = "xy"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a 2D xy grid to meshgrids for contour plotting.

    There may be a way to do this with a sparse meshgrid but here I create a
    non-sparse grid.

    Arguments
    ---------
    grid : A 2D grid in xy-plane (nz = 1). Can be a sparse grid.
    var : Name of variable in `grid` that is to be contoured. The `Z` array
        for mpl contour plotting.
    default : Default value for unpopulated parts of grid.
    plane : The plane of the 2D grid('xy', 'xz', 'yz')

    Returns
    -------
    Three meshgrids for x-coordinate, y-coordinate, and value (z-coordinate).

    """

    # Check plane argument.
    assert plane in {
        "xy",
        "xz",
        "yz",
    }, f"Plane must be one of: 'xy', 'xz', or 'yz' not {plane}"

    griddef: rmsp.GridDef = grid.griddef
    nx: int
    ny: int
    x_axis: str
    y_axis: str

    # Get parameters for orientation.
    x_axis, y_axis = plane[0], plane[1]
    if plane == "xy":
        nx = griddef.nu
        ny = griddef.nv
    elif plane == "xz":
        nx = griddef.nu
        ny = griddef.nz
    else:
        nx = griddef.nv
        ny = griddef.nz

    # Create a non-sparse grid.
    griddef.blockindices = np.arange(nx * ny)
    grid_full: rmsp.GridData = rmsp.GridData(griddef=griddef)

    # Copy variable from the sparse grid to the non-sparse grid.
    grid_full[var] = default
    grid_full.loc[grid.index, var] = grid[var]

    # TODO: Does this work for different planes?
    # # Get grid coordinates.
    grid_full[x_axis], grid_full[y_axis], _ = grid_full.coords()

    # Empty arrays to house the meshgrids.
    X: np.ndarray = np.ndarray(shape=(ny, nx), dtype=float, order="F")
    Y: np.ndarray = X.copy()
    Z: np.ndarray = X.copy()

    for idx, (iy, ix) in enumerate(itertools.product(range(ny), range(nx))):
        X[ny - 1 - iy][ix] = grid_full.loc[idx, x_axis]
        Y[ny - 1 - iy][ix] = grid_full.loc[idx, y_axis]
        Z[ny - 1 - iy][ix] = grid_full.loc[idx, var]
    return X, Y, Z


def grid_at_mesh(
    grid: rmsp.GridData,
    mesh: rmsp.Mesh,
    plane: str = "xy",
    plane_coord: float = 0.0,
) -> rmsp.GridData:
    """2D projection of grid on mesh surface.

    Projects nearest grid nodes to mesh.

    Arguments
    ---------
    grid: 3D grid.
    mesh: Open planar mesh.
    plane: Plane of 2d grid representation ('xy', 'xz', or 'yz').

    Returns
    -------
    A 2D grid representing the projection of the 3D grid onto the mesh."""

    # Check plane argument.
    assert plane in {
        "xy",
        "xz",
        "yz",
    }, f"plane must be one of: 'xy', 'xz', or 'yz' not {plane}"

    u_size: float = grid.griddef.usize
    v_size: float = grid.griddef.vsize
    z_size: float = grid.griddef.zsize

    # Get distances from mesh along each axis
    grid["u_dist"] = mesh.distance(grid, "x")
    grid["v_dist"] = mesh.distance(grid, "y")
    grid["z_dist"] = mesh.distance(grid, "z")
    # Get minimum distance across all axes.
    grid["dist"] = grid[["u_dist", "v_dist", "z_dist"]].min(axis=1)

    # Filter grid to those closest to mesh.
    filter_mesh: np.ndarray = (
        (grid["u_dist"] < u_size / 2)
        | (grid["v_dist"] < v_size / 2)
        | (grid["z_dist"] < z_size / 2)
    ).to_numpy()
    grid_intersect: rmsp.GridData = grid[filter_mesh]

    # Get axis perpendicular to plane axes.
    prp_ax: str = (set("xyz") - set(plane)).pop()
    # Remove duplicates in plane keeping highest (or most eastern, northern).
    pnt_intersect: rmsp.PointData = (
        grid_intersect.to_pointdata(x="x", y="y", z="z")
        .sort_values(prp_ax)
        .drop_duplicates(list(plane), "last")
    )

    # Convert to 2D grid (easier to plot than point data).
    pnt_intersect[prp_ax] = plane_coord
    
    return rmsp.GridData().from_centroids(
        pnt_intersect, u_size, v_size, z_size, 0.0
    )
