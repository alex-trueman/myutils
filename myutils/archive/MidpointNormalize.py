import numpy as np
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """Set specific end and mid-points for diverging colour scales.

    https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html"""

    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))