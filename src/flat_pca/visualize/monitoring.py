"""Control charts and scatter plots of the T² and Q statistics."""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

WITHIN_UCL_NAME = "within UCL"
EXCEEDS_UCL_NAME = "exceeds UCL"
EXCEEDS_UCL_COLOR = "#d62728"
UCL_LINE = {"color": EXCEEDS_UCL_COLOR, "dash": "dash", "width": 1}


def _split_traces(
    x: np.ndarray,
    y: np.ndarray,
    exceeds: np.ndarray,
    names: np.ndarray,
    hover: str,
    text: np.ndarray | None = None,
) -> list[go.Scatter]:
    """Return one marker trace within the limits and one above them.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates.
    exceeds : np.ndarray
        Boolean mask of the points above a control limit.
    names : np.ndarray
        Name of each point, carried as ``customdata``.
    hover : str
        Hover template of both traces.
    text : np.ndarray | None, optional
        Extra hover text of each point.

    Returns
    -------
    list[go.Scatter]
        The within-limit trace followed by the highlighted trace.
    """
    traces = []
    for mask, name, marker in (
        (~exceeds, WITHIN_UCL_NAME, {"size": 8}),
        (exceeds, EXCEEDS_UCL_NAME, {"size": 11, "color": EXCEEDS_UCL_COLOR, "symbol": "diamond"}),
    ):
        traces.append(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=name,
                customdata=names[mask],
                text=None if text is None else text[mask],
                marker=marker,
                hovertemplate=hover,
            )
        )
    return traces


def create_control_chart(
    values: np.ndarray | Sequence[float],
    *,
    ucl: float,
    labels: Sequence[str],
    y_name: str,
    order_values: Sequence[str] | None = None,
    order_name: str | None = None,
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
            xs, ys, ys > ucl, np.asarray(list(labels), dtype=object), hover, text
        )
    )
    figure.add_hline(
        y=ucl, line=UCL_LINE, annotation_text=f"UCL = {ucl:.4g}", annotation_position="top left"
    )
    x_title = "file order" if order_name is None else f"file order ({order_name})"
    return figure.update_layout(xaxis_title=x_title, yaxis_title=y_name)


def create_t2_q_scatter(
    t2: np.ndarray | Sequence[float],
    q: np.ndarray | Sequence[float],
    *,
    t2_ucl: float,
    q_ucl: float,
    labels: Sequence[str],
    t2_name: str = "T²",
    q_name: str = "Q",
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
    return figure.update_layout(xaxis_title=t2_name, yaxis_title=q_name)
