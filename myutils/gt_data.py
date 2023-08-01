import numpy as np
import pandas as pd


def gt_data(
    data,
    var,
    cog_var=None,
    cogs=np.arange(0.0, 3.1, 0.1),
    cell_size=["xinc", "yinc", "zinc"],
    density="density",
    tconvert=1.0e-3,
):
    """Calculate grade-tonnage data at cut-off grades.

    Alex Trueman, 2020-04

    Arguments
    ---------

    data : A DataFrame, rmsp.GridData, or rmsp.SubGridData object
        representing gridded spatial data. Should at least contain one
        grade/value column for evaluation. May also contain other
        columns as defined in the other function arguments. The grid may
        be 2D or 3D. **For SubGridData it will be necessary to provide
        each cell's dimension as column names in `cell_size`**.

    var : A string column name or list of string column names in `data`
        to be evaluated at cut-offs. Each is evaluated using the same
        cut-off column either defined with `cog_var` or as the first
        column name in `var`. Alternatively, a dictionary can be passed
        with column names as keys and unit conversion factors as values,
        Conversion factors convert the units of the returned metal
        values: e.g.:

            `var={"au_ppm": 1.0 / 31.1034768 / 1.0e3}` will convert the
            reported gold metal from grams to kilo-troy-ounces.

    cog_var : String column name in `data` to use as the cut-off
        variable. If this is `None` the first item/key in `var` will be
        used.

    cogs : 1D array-like or generator of cut-off grades.

    cell_size : Can be a tuple of numeric cell dimensions for one or
        more dimensions or a list of one or more cell dimension string
        column names in `data`. If one dimension, the values would be a
        cell surface area or volume. If four or more dimensions, the
        additional dimensions could be factors (0--1) for proportional
        cells, for example.

    density : A numeric constant or a string column name in `data` with
        variable density values. Used for tonnage and metal calculation
        along with the `cell_size` parameter.

    tconvert : A numeric tonnage conversion factor. The final tonnages
        are multiplied by this factor, e.g., `tconvert=1.0e-3` for
        kilotonnes. Note: this does not affect calculation of metal. To
        affect metal and grade calculations as well as tonnage:

            * A factor can be included with the `density` constant or in
              the values of the `density` column. 
            * A conversion/proportion column can be included as an
              additional dimension in the list of `cell_size` column
              names or cell size constants, e.g.:

                  `cell_size=["xinc", "yinc", "zinc", "prop"]`

    Returns
    -------

    A DataFrame of grade-tonnage-metal at cut-off grades. Grade columns
    have the same names as the inputs. Metal columns have the grade
    column names with a "_m" suffix.
    """

    # Make sure `var` is a dictionary.
    if isinstance(var, str):
        var = {var: 1}
    elif isinstance(var, list):
        var = {k: 1 for k in var}

    # Convert `cell_size` constants to columns.
    if all(isinstance(x, (int, float)) for x in cell_size):
        new_cols = {"col_" + str(i): value for i, value in enumerate(cell_size)}
        data = data.assign(**new_cols)
        cell_size = list(new_cols.keys())

    # Convert `density` constant to columns.
    if isinstance(density, (int, float)):
        data["density"] = density
        density = "density"

    # Create empty list of lists to store grade-tonnage data.
    gt_data = [[] for i in range(len(cogs))]
    # Loop by cut-off grade.
    # TODO: Something faster than a loop? Parallel this loop?
    for i, cog in enumerate(cogs):
        # Data for this cog.
        cog_data = data.loc[data[cog_var] >= cog, :]
        # Volume and tonnage.
        v = cog_data.loc[:, cell_size].prod(axis=1).sum()
        t = cog_data.loc[:, cell_size + [density]].prod(axis=1).sum()
        # Metal calculated for each variable in `var` dictionary.
        metal = []
        for v in var.keys():
            metal.append(
                cog_data.loc[:, cell_size + [density] + [v]].prod(axis=1).sum()
            )
        # Grades calculated from sum of tonnes and metal so that it is
        # tonnes weighted.
        grades = list(metal / t)

        # Apply unit conversion to metal (after re-calculation of grade).
        metal = [value * conv for value, conv in zip(metal, var.values())]

        # Write grade-tonnage data to list store.
        gt_data[i] = [cog_var, cog, t] + grades + metal

    # Create column names for final DataFrame.
    var_grade = [v for v in var.keys()]
    var_metal = [v + "_m" for v in var.keys()]
    cols = ["cog_var", "cog", "tonnes"] + var_grade + var_metal
    # Make the DataFrame.
    gt_df = pd.DataFrame(gt_data, columns=cols)

    # Apply conversion factor to tonnes after metal and grade calculations.
    gt_df["tonnes"] = gt_df["tonnes"] * tconvert

    return gt_df
