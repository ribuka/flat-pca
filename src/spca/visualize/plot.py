import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def _is_wavelength_column(column: str) -> bool:
    """Return whether a column name represents a numeric wavelength.

    Parameters
    ----------
    column : str
        DataFrame column name to evaluate.

    Returns
    -------
    bool
        ``True`` when the column name can be represented as a float.
    """
    try:
        float(column)
    except ValueError:
        return False
    return True


def create_spectra_heatmap(spectra: pl.DataFrame) -> go.Figure:
    """Create a spectral-intensity heatmap over time and wavelength.

    Parameters
    ----------
    spectra : pl.DataFrame
        Spectral data with a ``Time`` column and numeric wavelength-named
        columns. Other metadata columns are ignored.

    Returns
    -------
    go.Figure
        Heatmap with wavelength on the x-axis and time on the y-axis.
        Time zero is displayed at the bottom.
    """
    wavelength_columns = [
        column for column in spectra.columns if _is_wavelength_column(column)
    ]
    spectra_long = (
        spectra.unpivot(
            on=wavelength_columns,
            index="Time",
            variable_name="wavelength",
            value_name="intensity",
        )
        .with_columns(pl.col("wavelength").cast(pl.Float64))
    )
    heatmap_data = spectra_long.pivot(
        on="wavelength",
        index="Time",
        values="intensity",
        aggregate_function="first",
    ).sort("Time")

    return px.imshow(
        heatmap_data.drop("Time").to_numpy(),
        x=[float(column) for column in heatmap_data.columns[1:]],
        y=heatmap_data["Time"].to_list(),
        origin="lower",
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "wavelength", "y": "Time", "color": "intensity"},
    )
