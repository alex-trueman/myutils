"""Style a Dataframe in a standard way."""


def style_df(df, str_cols, hide_cols, sigfigs=2):
    """Standard styling for Dataframe.

    Format string, integer, and float columns in a standard way.
    The inde is hidden.

    Parameters
    ----------
    df : The dataframe to be styled.
    str_cols : Object-type columns and their headings will be left-aligned.
    hide_cols : Columns to be hideen from display.
    sig_figs : Number of significant figures for float columns.

    Return
    ------
    Styled dataframe.

    """
    return (
        df.style.hide_index()
        .hide_columns(hide_cols)
        .set_properties(str_cols, **{"text-align": "left"})
        .set_table_styles(
            {
                col: [{"selector": "", "props": "text-align: left"}]
                for col in str_cols
            }
        )
        .format({col: "{:,d}" for col in df.select_dtypes(int).columns})
        .format(
            {
                col: f"{{:,.{sigfigs}g}}"
                for col in df.select_dtypes(float).columns
            }
        )
    )
