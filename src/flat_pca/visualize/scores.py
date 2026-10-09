"""Scatter plots of PCA scores and partial score trajectories."""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl

from .component_plane import apply_component_plane_layout
from .marker_color import MISSING_COLOR, category_color, color_groups, continuous_marker


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
            marker |= continuous_marker(color)
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


def _scale_color(value: float, marker: dict[str, object]) -> str:
    """Return the color a continuous marker gives one value.

    Parameters
    ----------
    value : float
        Finite color value.
    marker : dict[str, object]
        Marker settings from ``continuous_marker`` for a series with a
        finite value, so it has a color scale.

    Returns
    -------
    str
        Color of ``value`` on the marker's color scale.
    """
    low, high = float(marker["cmin"]), float(marker["cmax"])  # type: ignore[arg-type]
    fraction = (value - low) / (high - low) if high > low else 0.5
    return plotly.colors.sample_colorscale(str(marker["colorscale"]), [fraction])[0]


def _trajectory_styles(
    color: pl.Series | None, point_counts: Sequence[int]
) -> list[dict[str, object]]:
    """Return the ``go.Scatter`` settings coloring each trajectory.

    Parameters
    ----------
    color : pl.Series | None
        Value coloring each trajectory; see
        ``create_partial_score_trajectories``.
    point_counts : Sequence[int]
        Number of points of each trajectory.

    Returns
    -------
    list[dict[str, object]]
        ``line``, ``marker``, and ``showlegend`` of each trajectory, plus
        ``name`` and ``legendgroup`` for a categorical color value.
    """
    if color is None:
        single = category_color(0)
        return [
            {"line": {"color": single}, "marker": {"color": single}, "showlegend": False}
            for _ in point_counts
        ]
    if color.dtype.is_numeric():
        values = color.cast(pl.Float64).to_numpy()
        scaled = np.flatnonzero(np.isfinite(values))
        styles: list[dict[str, object]] = []
        for index, count in enumerate(point_counts):
            if not np.isfinite(values[index]):
                missing = {"color": MISSING_COLOR}
                styles.append({"line": missing, "marker": missing, "showlegend": False})
                continue
            # One row per point, so the color scale colors every point. The
            # first trajectory with a value draws the shared color bar.
            rows = np.full(count, index, dtype=np.intp)
            marker = continuous_marker(color, rows, showscale=index == int(scaled[0]))
            styles.append(
                {
                    "line": {"color": _scale_color(float(values[index]), marker)},
                    "marker": marker,
                    "showlegend": False,
                }
            )
        return styles
    styles = [{} for _ in point_counts]
    for group, (value, rows) in enumerate(color_groups(color).items()):
        shade = category_color(group)
        for position, row in enumerate(rows.tolist()):
            styles[row] = {
                "name": value,
                "legendgroup": value,
                "showlegend": position == 0,
                "line": {"color": shade},
                "marker": {"color": shade},
            }
    return styles


def create_partial_score_trajectories(
    trajectories: Mapping[
        str,
        tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float], Sequence[str]],
    ],
    *,
    x_name: str,
    y_name: str,
    color: pl.Series | None = None,
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
    color : pl.Series | None, optional
        Value coloring each trajectory, in mapping order, titled by the
        series name. A numeric series colors the lines and points with a
        continuous color scale; any other series gives each value one color
        and one legend entry (missing values share one). By default every
        trajectory has the same color and there is no legend.

    Returns
    -------
    go.Figure
        One line trace per trajectory.
    """
    point_counts = [len(point_labels) for _, _, point_labels in trajectories.values()]
    styles = _trajectory_styles(color, point_counts)
    traces = []
    for (label, (x, y, point_labels)), style in zip(
        trajectories.items(), styles, strict=True
    ):
        xs = np.asarray(x, dtype=np.float64)
        sizes = np.full(xs.size, 5.0)
        if sizes.size:
            sizes[-1] = 12.0
        traces.append(
            go.Scatter(
                x=xs,
                y=np.asarray(y, dtype=np.float64),
                mode="lines+markers",
                name=style.pop("name", label),
                text=list(point_labels),
                marker={"size": sizes} | style.pop("marker"),
                hovertemplate=(
                    f"{label}<br>%{{text}}<br>{x_name}=%{{x:.4g}}<br>"
                    f"{y_name}=%{{y:.4g}}<extra></extra>"
                ),
                **style,
            )
        )
    figure = go.Figure(traces).update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        legend_title_text=None if color is None else color.name,
    )
    return apply_component_plane_layout(figure)
