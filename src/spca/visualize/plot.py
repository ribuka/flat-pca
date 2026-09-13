import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def create_spectra_heatmap(spectra: pl.DataFrame) -> go.Figure:
    """Create a spectral-intensity heatmap over time and wavelength.

    Parameters
    ----------
    spectra : pl.DataFrame
        Spectral data with a ``Time`` column and wavelength-named columns.

    Returns
    -------
    go.Figure
        Heatmap with wavelength on the x-axis and time on the y-axis.
        Time zero is displayed at the bottom.
    """
    spectra_long = (
        spectra.unpivot(
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
