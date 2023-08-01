from typing import Tuple


def set_fig_size(width: float = 620, fraction: float = 1, subplot=None, square: bool = False) -> Tuple:
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Modified from: https://jwalton.info//Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: Width of figure in points.
    fraction: Fraction of the width which you wish the figure to occupy.
    subplot: Two-item list with n_rows and n_cols.
    square: make width and height the same or use golden ratio.

    Returns
    -------
    fig_dim: Dimensions of figure in inches.

    Example
    -------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=set_fig_size(subplot=[2, 1]))
    """
    if subplot is None:
        subplot = [1, 1]

    # Width of figure.
    fig_width_pt = width * fraction

    # Convert from points to inches.
    inches_per_pt = 1 / 72.27

    # Ratio to set aesthetic figure height.
    if square:
        ratio = 1
    else:
        ratio = (5 ** .5 - 1) / 2

    # Figure width in inches.
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches.
    fig_height_in = fig_width_in * ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
