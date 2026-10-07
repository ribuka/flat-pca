"""Scatter plots of PCA loadings aggregated by wavelength."""

import plotly.graph_objects as go
import polars as pl

from .component_plane import apply_component_plane_layout


def create_loading_scatter(
    loadings: pl.DataFrame,
    *,
    x: str,
    y: str,
    wavelength: str = "wavelength",
) -> go.Figure:
    """Create a scatter plot of two loading columns, one point per wavelength.

    Parameters
    ----------
    loadings : pl.DataFrame
        One row per wavelength with the ``wavelength``, ``x``, and ``y``
        columns, such as two components of
        ``aggregate_loadings_by_wavelength`` side by side.
    x : str
        Loading column of the horizontal axis.
    y : str
        Loading column of the vertical axis.
    wavelength : str, default ``"wavelength"``
        Column of the wavelengths, which color the points.

    Returns
    -------
    go.Figure
        Loading scatter plot with a wavelength color bar.
    """
    wavelengths = loadings[wavelength].cast(pl.Float64).to_numpy()
    figure = go.Figure(
        go.Scatter(
            x=loadings[x].cast(pl.Float64).to_numpy(),
            y=loadings[y].cast(pl.Float64).to_numpy(),
            mode="markers",
            customdata=wavelengths,
            marker={
                "size": 7,
                "color": wavelengths,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": {"text": wavelength}},
            },
            hovertemplate=(
                f"{wavelength}=%{{customdata:g}}<br>{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    return apply_component_plane_layout(figure.update_layout(xaxis_title=x, yaxis_title=y))
