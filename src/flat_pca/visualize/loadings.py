"""Scatter plots of PCA loadings aggregated by a feature key."""

import plotly.graph_objects as go
import polars as pl

from .component_plane import apply_component_plane_layout


def create_loading_scatter(
    loadings: pl.DataFrame,
    *,
    x: str,
    y: str,
    color: str = "wavelength",
) -> go.Figure:
    """Create a scatter plot of two loading columns, one point per key value.

    Parameters
    ----------
    loadings : pl.DataFrame
        One row per key value with the ``color``, ``x``, and ``y`` columns,
        such as two components of ``aggregate_loadings`` side by side.
    x : str
        Loading column of the horizontal axis.
    y : str
        Loading column of the vertical axis.
    color : str, default ``"wavelength"``
        Column of the key values the loadings are aggregated by, such as
        ``"wavelength"`` or ``"StepTime"``. They color the points and are
        shown on hover.

    Returns
    -------
    go.Figure
        Loading scatter plot with a color bar of the ``color`` column.
    """
    keys = loadings[color].cast(pl.Float64).to_numpy()
    figure = go.Figure(
        go.Scatter(
            x=loadings[x].cast(pl.Float64).to_numpy(),
            y=loadings[y].cast(pl.Float64).to_numpy(),
            mode="markers",
            customdata=keys,
            marker={
                "size": 7,
                "color": keys,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": {"text": color}},
            },
            hovertemplate=(
                f"{color}=%{{customdata:g}}<br>{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    return apply_component_plane_layout(figure.update_layout(xaxis_title=x, yaxis_title=y))
