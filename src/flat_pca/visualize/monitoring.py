"""Control charts and scatter plots of the T² and Q statistics."""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .marker_color import category_color, color_groups, continuous_marker_groups

UCL_COLOR = "#d62728"
UCL_LINE = {"color": UCL_COLOR, "dash": "dash", "width": 1}
MARKER = {"size": 8}


def _marker_trace(
    rows: np.ndarray | slice,
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
    rows : np.ndarray | slice
        Positions of the points of the trace.
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


def _marker_traces(
    x: np.ndarray,
    y: np.ndarray,
    names: np.ndarray,
    hover: str,
    text: np.ndarray | None = None,
    color: pl.Series | None = None,
    rows: np.ndarray | None = None,
) -> list[go.Scatter]:
    """Return the marker traces of the points, colored like the score scatter plot.

    Every point is drawn with the same marker, whether or not it lies above
    a control limit.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates.
    names : np.ndarray
        Name of each point, carried as ``customdata``.
    hover : str
        Hover template of all traces.
    text : np.ndarray | None, optional
        Extra hover text of each point.
    color : pl.Series | None, optional
        Value coloring each point, titled by the series name. A numeric
        series is drawn with a continuous color scale, its missing points
        faint in a trace behind the others; any other series draws one trace
        per value, named by the value.
    rows : np.ndarray | None, optional
        Ascending positions of the points drawn; all points by default. The
        colors still follow all of ``color`` (the range of a numeric one,
        the order of the values of a categorical one), so figures drawing
        different points of the same files color them alike.

    Returns
    -------
    list[go.Scatter]
        One trace of the drawn points without ``color``; with a numeric
        one, a trace of the drawn points missing it, then one of those with
        a value; one trace of each value with drawn points with a categorical
        one.
    """
    drawn = np.arange(y.size) if rows is None else np.asarray(rows, dtype=np.intp)
    if color is None:
        return [_marker_trace(drawn, x, y, names, hover, text, marker=MARKER)]
    if color.dtype.is_numeric():
        # The color bar explains the colors, so the split traces need no legend.
        return [
            _marker_trace(
                group, x, y, names, hover, text, marker=MARKER | marker, showlegend=False
            )
            for group, marker in continuous_marker_groups(color, drawn)
        ]
    is_drawn = np.zeros(y.size, dtype=bool)
    is_drawn[drawn] = True
    return [
        _marker_trace(
            group[is_drawn[group]],
            x,
            y,
            names,
            hover,
            text,
            name=value,
            marker=MARKER | {"color": category_color(index)},
        )
        for index, (value, group) in enumerate(color_groups(color).items())
        if is_drawn[group].any()
    ]


def _color_layout(color: pl.Series | None) -> dict[str, object]:
    """Return the legend layout of a figure colored by ``color``.

    Parameters
    ----------
    color : pl.Series | None
        Value coloring the points, or ``None``.

    Returns
    -------
    dict[str, object]
        Layout properties: a categorical color titles the legend by its
        column; otherwise there is no legend to lay out.
    """
    if color is None or color.dtype.is_numeric():
        return {}
    return {"legend_title_text": color.name}


def _axis_values(x: pl.Series) -> np.ndarray:
    """Return the values of a numeric or temporal series as plot coordinates.

    Parameters
    ----------
    x : pl.Series
        Numeric, date, or datetime values.

    Returns
    -------
    np.ndarray
        ``float64`` values of a numeric series; Python ``date`` or
        ``datetime`` objects otherwise, which Plotly draws on a date axis.
    """
    if x.dtype.is_numeric():
        return x.cast(pl.Float64).to_numpy()
    return np.asarray(x.to_list(), dtype=object)


def create_control_chart(
    values: np.ndarray | Sequence[float],
    *,
    ucl: float,
    labels: Sequence[str],
    y_name: str,
    x: pl.Series | None = None,
    x_text: Sequence[str] | None = None,
    x_name: str | None = None,
    color: pl.Series | None = None,
    rows: np.ndarray | None = None,
) -> go.Figure:
    """Create a control chart of one statistic.

    Without ``x``, the points are drawn at the positions ``1..n`` in the
    order given; with ``x``, at its values on a numeric or date axis. A
    dashed line marks the upper control limit. Each point carries its label
    as ``customdata``, so a click handler can tell which sample was chosen.

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
    x : pl.Series | None, optional
        Numeric, date, or datetime horizontal position of each point,
        missing only where a point is not drawn (see ``rows``). ``None``
        for the positions ``1..n``.
    x_text : Sequence[str] | None, optional
        Text of the horizontal-axis value of each point, shown in the hover
        text.
    x_name : str | None, optional
        Name of the horizontal-axis column. It titles the axis, as
        ``file order (<x_name>)`` without ``x``.
    color : pl.Series | None, optional
        Value coloring each point, in display order; see ``_marker_traces``.
    rows : np.ndarray | None, optional
        Ascending positions of the points drawn, such as those with an
        ``x`` value; all points by default. Every argument still holds a
        value for every point, so the colors match a figure of all points.

    Returns
    -------
    go.Figure
        Control chart.
    """
    ys = np.asarray(values, dtype=np.float64)
    xs = np.arange(1, ys.size + 1) if x is None else _axis_values(x)
    text = None if x_text is None else np.asarray(list(x_text), dtype=object)
    x_line = "" if text is None else f"{x_name or 'x'}=%{{text}}<br>"
    hover = "%{customdata}<br>" + x_line + f"{y_name}=%{{y:.4g}}<extra></extra>"
    figure = go.Figure(
        _marker_traces(
            xs, ys, np.asarray(list(labels), dtype=object), hover, text, color, rows
        )
    )
    figure.add_hline(
        y=ucl, line=UCL_LINE, annotation_text=f"UCL = {ucl:.4g}", annotation_position="top left"
    )
    if x is not None:
        x_title = x_name
    else:
        x_title = "file order" if x_name is None else f"file order ({x_name})"
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
        Value coloring each point; see ``_marker_traces``.

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
        _marker_traces(xs, ys, np.asarray(list(labels), dtype=object), hover, color=color)
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
