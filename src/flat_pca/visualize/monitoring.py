"""Control charts and scatter plots of the T² and Q statistics."""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .marker_color import category_color, color_groups, continuous_marker

WITHIN_UCL_NAME = "within UCL"
EXCEEDS_UCL_NAME = "exceeds UCL"
EXCEEDS_UCL_COLOR = "#d62728"
UCL_LINE = {"color": EXCEEDS_UCL_COLOR, "dash": "dash", "width": 1}
WITHIN_MARKER = {"size": 8}
EXCEEDS_MARKER = {"size": 11, "symbol": "diamond"}
# A colored point above a limit keeps its color and gets this outline.
EXCEEDS_OUTLINE = {"line": {"color": EXCEEDS_UCL_COLOR, "width": 2}}


def _marker_trace(
    rows: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    names: np.ndarray,
    hover: str,
    text: np.ndarray | None,
    **trace: object,
) -> go.Scatter:
    """Return a marker trace of some of the points.

    Parameters
    ----------
    rows : np.ndarray
        Boolean mask or positions of the points of the trace.
    x, y : np.ndarray
        Coordinates of all points.
    names : np.ndarray
        Name of each point, carried as ``customdata``.
    hover : str
        Hover template.
    text : np.ndarray | None
        Extra hover text of each point.
    **trace : object
        Further ``go.Scatter`` properties, such as ``name`` and ``marker``.

    Returns
    -------
    go.Scatter
        Marker trace of the chosen points.
    """
    return go.Scatter(
        x=x[rows],
        y=y[rows],
        mode="markers",
        customdata=names[rows],
        text=None if text is None else text[rows],
        hovertemplate=hover,
        **trace,
    )


def _split_traces(
    x: np.ndarray,
    y: np.ndarray,
    exceeds: np.ndarray,
    names: np.ndarray,
    hover: str,
    text: np.ndarray | None = None,
    color: pl.Series | None = None,
) -> list[go.Scatter]:
    """Return marker traces that highlight the points above a limit.

    Without ``color``, the points within the limits form one trace and the
    points above them a red, larger diamond trace. With ``color``, every
    point is colored by its value like the score scatter plot, and the points
    above a limit are larger diamonds outlined in red.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates.
    exceeds : np.ndarray
        Boolean mask of the points above a control limit.
    names : np.ndarray
        Name of each point, carried as ``customdata``.
    hover : str
        Hover template of all traces.
    text : np.ndarray | None, optional
        Extra hover text of each point.
    color : pl.Series | None, optional
        Value coloring each point, titled by the series name. A numeric
        series is drawn with a continuous color scale shared by both traces;
        any other series draws a within-limit and an above-limit trace per
        value, grouped in the legend by value.

    Returns
    -------
    list[go.Scatter]
        Without ``color`` or with a numeric one, the within-limit trace
        followed by the highlighted trace; with a categorical one, those two
        traces of each value in turn.
    """
    if color is None:
        return [
            _marker_trace(~exceeds, x, y, names, hover, text, name=WITHIN_UCL_NAME, marker=WITHIN_MARKER),
            _marker_trace(
                exceeds,
                x,
                y,
                names,
                hover,
                text,
                name=EXCEEDS_UCL_NAME,
                marker=EXCEEDS_MARKER | {"color": EXCEEDS_UCL_COLOR},
            ),
        ]
    if color.dtype.is_numeric():
        within = np.flatnonzero(~exceeds)
        above = np.flatnonzero(exceeds)
        # The one color bar goes on a trace holding a finite color value, as
        # Plotly.js cannot draw the bar of a trace whose values are all missing.
        finite = np.isfinite(color.cast(pl.Float64).to_numpy())
        within_scale = bool(finite[within].any())
        return [
            _marker_trace(
                within,
                x,
                y,
                names,
                hover,
                text,
                name=WITHIN_UCL_NAME,
                marker=WITHIN_MARKER | continuous_marker(color, within, showscale=within_scale),
            ),
            _marker_trace(
                above,
                x,
                y,
                names,
                hover,
                text,
                name=EXCEEDS_UCL_NAME,
                marker=EXCEEDS_MARKER
                | continuous_marker(color, above, showscale=not within_scale)
                | EXCEEDS_OUTLINE,
            ),
        ]
    traces = []
    for index, (value, rows) in enumerate(color_groups(color).items()):
        fill = {"color": category_color(index)}
        traces += [
            _marker_trace(
                rows[~exceeds[rows]],
                x,
                y,
                names,
                hover,
                text,
                name=value,
                legendgroup=value,
                marker=WITHIN_MARKER | fill,
            ),
            _marker_trace(
                rows[exceeds[rows]],
                x,
                y,
                names,
                hover,
                text,
                name=f"{value} ({EXCEEDS_UCL_NAME})",
                legendgroup=value,
                showlegend=bool(exceeds[rows].any()),
                marker=EXCEEDS_MARKER | fill | EXCEEDS_OUTLINE,
            ),
        ]
    return traces


def _color_layout(color: pl.Series | None) -> dict[str, object]:
    """Return the legend layout of a figure colored by ``color``.

    Parameters
    ----------
    color : pl.Series | None
        Value coloring the points, or ``None``.

    Returns
    -------
    dict[str, object]
        Layout properties: the legend is titled by the color column, and
        with a numeric one it lies above the plot, clear of the color bar.
    """
    if color is None:
        return {}
    if color.dtype.is_numeric():
        return {
            "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"}
        }
    return {"legend_title_text": color.name}


def create_control_chart(
    values: np.ndarray | Sequence[float],
    *,
    ucl: float,
    labels: Sequence[str],
    y_name: str,
    order_values: Sequence[str] | None = None,
    order_name: str | None = None,
    color: pl.Series | None = None,
) -> go.Figure:
    """Create a control chart of one statistic in the given file order.

    The points are drawn at positions ``1..n`` in the order given, with a
    dashed line at the upper control limit. Points above the limit are
    drawn in a separate, highlighted trace. Each point carries its label as
    ``customdata``, so a click handler can tell which sample was chosen.

    Parameters
    ----------
    values : np.ndarray | Sequence[float]
        Statistic of each sample, in display order.
    ucl : float
        Upper control limit.
    labels : Sequence[str]
        Name of each point in the hover text and ``customdata``.
    y_name : str
        Label of the vertical axis, such as ``"T²"``.
    order_values : Sequence[str] | None, optional
        Text of the ordering value of each point, shown in the hover text.
    order_name : str | None, optional
        Name of the ordering column; the horizontal axis is titled with it.
    color : pl.Series | None, optional
        Value coloring each point, in display order; see ``_split_traces``.

    Returns
    -------
    go.Figure
        Control chart.
    """
    ys = np.asarray(values, dtype=np.float64)
    xs = np.arange(1, ys.size + 1)
    text = None if order_values is None else np.asarray(list(order_values), dtype=object)
    order_line = "" if text is None else f"{order_name or 'order'}=%{{text}}<br>"
    hover = (
        "%{customdata}<br>" + order_line + f"{y_name}=%{{y:.4g}}<extra></extra>"
    )
    figure = go.Figure(
        _split_traces(
            xs, ys, ys > ucl, np.asarray(list(labels), dtype=object), hover, text, color
        )
    )
    figure.add_hline(
        y=ucl, line=UCL_LINE, annotation_text=f"UCL = {ucl:.4g}", annotation_position="top left"
    )
    x_title = "file order" if order_name is None else f"file order ({order_name})"
    return figure.update_layout(xaxis_title=x_title, yaxis_title=y_name, **_color_layout(color))


def create_t2_q_scatter(
    t2: np.ndarray | Sequence[float],
    q: np.ndarray | Sequence[float],
    *,
    t2_ucl: float,
    q_ucl: float,
    labels: Sequence[str],
    t2_name: str = "T²",
    q_name: str = "Q",
    color: pl.Series | None = None,
) -> go.Figure:
    """Create a scatter plot of T² against Q with both control limits.

    Points above either limit are drawn in a separate, highlighted trace.
    Each point carries its label as ``customdata``.

    Parameters
    ----------
    t2 : np.ndarray | Sequence[float]
        T² of each sample (horizontal axis).
    q : np.ndarray | Sequence[float]
        Q of each sample (vertical axis).
    t2_ucl : float
        Upper control limit of T², drawn as a vertical line.
    q_ucl : float
        Upper control limit of Q, drawn as a horizontal line.
    labels : Sequence[str]
        Name of each point in the hover text and ``customdata``.
    t2_name : str, default "T²"
        Label of the horizontal axis.
    q_name : str, default "Q"
        Label of the vertical axis.
    color : pl.Series | None, optional
        Value coloring each point; see ``_split_traces``.

    Returns
    -------
    go.Figure
        T² × Q scatter plot.
    """
    xs = np.asarray(t2, dtype=np.float64)
    ys = np.asarray(q, dtype=np.float64)
    hover = (
        "%{customdata}<br>" + f"{t2_name}=%{{x:.4g}}<br>{q_name}=%{{y:.4g}}<extra></extra>"
    )
    figure = go.Figure(
        _split_traces(
            xs,
            ys,
            (xs > t2_ucl) | (ys > q_ucl),
            np.asarray(list(labels), dtype=object),
            hover,
            color=color,
        )
    )
    figure.add_vline(
        x=t2_ucl,
        line=UCL_LINE,
        annotation_text=f"{t2_name} UCL = {t2_ucl:.4g}",
        annotation_position="top right",
    )
    figure.add_hline(
        y=q_ucl,
        line=UCL_LINE,
        annotation_text=f"{q_name} UCL = {q_ucl:.4g}",
        annotation_position="top left",
    )
    return figure.update_layout(xaxis_title=t2_name, yaxis_title=q_name, **_color_layout(color))
