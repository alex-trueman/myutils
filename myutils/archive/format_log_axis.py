import matplotlib.ticker as ticker


def format_log_axis(axis):
    """Format log axis to plain numbers.
    https://stackoverflow.com/a/33213196/4516267
        
    Parameters
    ----------
    axis : Matplotlib axis (e.g., ax.xaxis)
    
    Returns
    -------
    None
    """
    axis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "{:g}".format(x)))
    return None
