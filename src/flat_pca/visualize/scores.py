"""Scatter plots of PCA scores and partial score trajectories."""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl

MISSING_LABEL = "(missing)"


def _color_groups(color: pl.Series) -> dict[str, np.ndarray]:
    """Group row positions by the text of a categorical color value.

    Parameters
    ----------
    color : pl.Series
        Color value of each row.

    Returns
    -------
    dict[str, np.ndarray]
        Row positions keyed by value text, in order of first appearance;
        missing values are grouped as ``MISSING_LABEL``.
    """
    labels = color.cast(pl.String).fill_null(MISSING_LABEL).to_list()
    groups: dict[str, list[int]] = {}
    for position, label in enumerate(labels):
        groups.setdefault(label, []).append(position)
    return {label: np.asarray(rows, dtype=np.intp) for label, rows in groups.items()}


def create_score_scatter(
    scores: pl.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    color: str | None = None,
) -> go.Figure:
    """Create a scatter plot of two score columns, one point per row.

    Each point carries its ``label`` value as ``customdata``, so a click
    handler can tell which row was chosen.

    Parameters
    ----------
    scores : pl.DataFrame
        One row per sample with the ``x``, ``y``, ``label``, and ``color``
        columns.
    x : str
        Score column of the horizontal axis.
    y : str
        Score column of the vertical axis.
    label : str
        Column naming each point in the hover text and ``customdata``.
    color : str | None, optional
        Column coloring the points. A numeric column is drawn with a
        continuous color scale; any other column draws one trace per value.
        By default all points share one trace.

    Returns
    -------
    go.Figure
        Score scatter plot.
    """
    labels = scores[label].cast(pl.String).to_numpy()
    xs = scores[x].cast(pl.Float64).to_numpy()
    ys = scores[y].cast(pl.Float64).to_numpy()
    hover = "%{customdata}<br>" + f"{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra></extra>"
    if color is None or scores[color].dtype.is_numeric():
        marker: dict[str, object] = {"size": 9}
        if color is not None:
            marker |= {
                "color": scores[color].cast(pl.Float64).to_numpy(),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": {"text": color}},
            }
        traces = [
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                customdata=labels,
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
                customdata=labels[rows],
                marker={"size": 9},
                hovertemplate=hover,
            )
            for value, rows in _color_groups(scores[color]).items()
        ]
    return go.Figure(traces).update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color,
    )


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
    return go.Figure(traces).update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
    )
