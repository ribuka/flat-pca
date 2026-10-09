"""Line plots of spectral values along one axis of a heatmap."""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

# Height of the plot area (frame) of a trend, in pixels, whatever the number
# of legend rows.
TREND_FRAME_HEIGHT = 320
# Margins around the plot area, in pixels: the top one holds the title, the
# legend rows above the plot area add to it.
TREND_MARGIN = {"t": 40, "b": 56, "l": 64, "r": 24}


def trend_height(frame_height: float, legend_height: float = 0.0) -> float:
    """Return the figure height that gives a trend a plot area of ``frame_height``.

    Parameters
    ----------
    frame_height : float
        Height of the plot area in pixels.
    legend_height : float, default 0.0
        Height of the legend above the plot area in pixels.

    Returns
    -------
    float
        Top margin, legend, plot area, and bottom margin heights summed.
    """
    return TREND_MARGIN["t"] + legend_height + frame_height + TREND_MARGIN["b"]


def create_trend(
    lines: Mapping[str, tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
    *,
    x_name: str,
    y_name: str = "intensity",
    title: str | None = None,
    frame_height: float = TREND_FRAME_HEIGHT,
) -> go.Figure:
    """Create a line plot overlaying one trace per labeled series.

    The horizontal legend lies above the plot area, so long labels do not
    narrow it. The figure height leaves no room for the legend; the page
    adds the height of the drawn legend to the height and the top margin,
    so the plot area keeps ``frame_height`` however many rows the legend has.

    Parameters
    ----------
    lines : Mapping[str, tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]]
        ``(x, y)`` values keyed by the trace label, drawn in mapping order.
        NaN values leave gaps in their trace.
    x_name : str
        Label of the x-axis, such as ``"StepTime"`` or ``"wavelength"``.
    y_name : str, default ``"intensity"``
        Label of the y-axis.
    title : str | None, optional
        Figure title, by default none.
    frame_height : float, default ``TREND_FRAME_HEIGHT``
        Height of the plot area in pixels.

    Returns
    -------
    go.Figure
        One line trace per entry of ``lines``, with a legend.
    """
    figure = go.Figure(
        [
            go.Scatter(
                x=np.asarray(x, dtype=np.float64),
                y=np.asarray(y, dtype=np.float64),
                mode="lines+markers",
                name=label,
            )
            for label, (x, y) in lines.items()
        ]
    )
    return figure.update_layout(
        title={"text": title, "yref": "container", "y": 1, "yanchor": "top", "pad": {"t": 12}},
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1, "yanchor": "bottom"},
        margin=TREND_MARGIN,
        height=trend_height(frame_height),
        # Keep the width following the container: a set height alone fixes it.
        autosize=True,
    )
