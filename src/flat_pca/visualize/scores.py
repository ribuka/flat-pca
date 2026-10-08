"""Scatter plots of PCA scores and partial score trajectories."""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .component_plane import apply_component_plane_layout
from .marker_color import color_groups, continuous_marker


def create_score_scatter(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    *,
    labels: Sequence[str],
    x_name: str,
    y_name: str,
    color: pl.Series | None = None,
) -> go.Figure:
    """Create a scatter plot of two components' scores, one point per sample.

    Each point carries its label as ``customdata``, so a click handler can
    tell which sample was chosen.

    Parameters
    ----------
    x : np.ndarray | Sequence[float]
        Scores of the horizontal axis.
    y : np.ndarray | Sequence[float]
        Scores of the vertical axis.
    labels : Sequence[str]
        Name of each point in the hover text and ``customdata``.
    x_name : str
        Label of the horizontal axis, such as ``"PC1"``.
    y_name : str
        Label of the vertical axis.
    color : pl.Series | None, optional
        Value coloring each point, titled by the series name. A numeric
        series is drawn with a continuous color scale; any other series
        draws one trace per value. By default all points share one trace.

    Returns
    -------
    go.Figure
        Score scatter plot.
    """
    names = np.asarray(list(labels), dtype=object)
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    hover = "%{customdata}<br>" + f"{x_name}=%{{x:.4g}}<br>{y_name}=%{{y:.4g}}<extra></extra>"
    if color is None or color.dtype.is_numeric():
        marker: dict[str, object] = {"size": 9}
        if color is not None:
            marker |= continuous_marker(color) | {"showscale": True}
        traces = [
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                customdata=names,
                marker=marker,
                hovertemplate=hover,
                showlegend=False,
            )
        ]
    else:
        traces = [
            go.Scatter(
                x=xs[rows],
                y=ys[rows],
                mode="markers",
                name=value,
                customdata=names[rows],
                marker={"size": 9},
                hovertemplate=hover,
            )
            for value, rows in color_groups(color).items()
        ]
    figure = go.Figure(traces).update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        legend_title_text=None if color is None else color.name,
    )
    return apply_component_plane_layout(figure)


def create_partial_score_trajectories(
    trajectories: Mapping[
        str,
        tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float], Sequence[str]],
    ],
    *,
    x_name: str,
    y_name: str,
) -> go.Figure:
    """Create a plot of partial score trajectories in a score plane.

    Parameters
    ----------
    trajectories : Mapping[str, tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float], Sequence[str]]]
        ``(x, y, point_labels)`` keyed by trace label, drawn in mapping
        order. ``point_labels`` names each point in the hover text, such as
        its ``(Step, Sequence, StepTime)``. The last point of each
        trajectory is marked larger.
    x_name : str
        Label of the horizontal axis.
    y_name : str
        Label of the vertical axis.

    Returns
    -------
    go.Figure
        One line trace per trajectory, with a legend.
    """
    traces = []
    for label, (x, y, point_labels) in trajectories.items():
        xs = np.asarray(x, dtype=np.float64)
        sizes = np.full(xs.size, 5.0)
        if sizes.size:
            sizes[-1] = 12.0
        traces.append(
            go.Scatter(
                x=xs,
                y=np.asarray(y, dtype=np.float64),
                mode="lines+markers",
                name=label,
                text=list(point_labels),
                marker={"size": sizes},
                hovertemplate=(
                    f"{label}<br>%{{text}}<br>{x_name}=%{{x:.4g}}<br>"
                    f"{y_name}=%{{y:.4g}}<extra></extra>"
                ),
            )
        )
    figure = go.Figure(traces).update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
    )
    return apply_component_plane_layout(figure)
