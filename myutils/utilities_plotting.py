"""Plotting utilities."""

__author__ = "Alex M Trueman"


def format_cbar_label(
    axes, rotation=90.0, verticle_alignment="center", **kwargs
):
    """Format colorbar labels.

    This works for axes returned from rmsp sectionplots, ImageGrid, and
    nested_colorbar, where the colorbar object is included. Also from
    rmsp ImageGrid.

    Parameters
    ----------
    axes : axes returned from sectionplots with colorbar object.
    rotation : label rotation.
    verticle_alignment : label vertical alignment.
    kwargs : other matplotlib Text properties (e.g., fontsize).

    Returns
    -------
    The input axes.
    """
    axes.cax.set_yticklabels(
        axes.cax.get_yticklabels(),
        rotation=rotation,
        va=verticle_alignment,
        **kwargs
    )
    return axes
