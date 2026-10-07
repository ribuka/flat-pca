"""Line plots of spectral values along one axis of a heatmap."""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


def create_trend(
    lines: Mapping[str, tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
    *,
    x_name: str,
    y_name: str = "intensity",
    title: str | None = None,
) -> go.Figure:
    """Create a line plot overlaying one trace per labeled series.

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
        title=title,
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
    )
