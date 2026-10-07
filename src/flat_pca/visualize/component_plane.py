"""Layout shared by plots of component m against component n."""

import numpy as np
import plotly.graph_objects as go

ZERO_LINE_COLOR = "#888"
RANGE_PADDING = 0.05


def _padded_range(values: list[np.ndarray]) -> tuple[float, float]:
    """Return the range of values with padding on both sides.

    Parameters
    ----------
    values : list[np.ndarray]
        Values of each trace; non-finite values are ignored.

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` widened by ``RANGE_PADDING`` of the span on each
        side; ``(-1.0, 1.0)`` without finite values, and a unit span around
        the value when all values are equal.
    """
    finite = np.concatenate([np.ravel(value) for value in values] or [np.empty(0)])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        return low - 0.5, high + 0.5
    padding = (high - low) * RANGE_PADDING
    return low - padding, high + padding


def apply_component_plane_layout(figure: go.Figure) -> go.Figure:
    """Draw a component plane in a square frame with emphasized zero lines.

    The axis ranges follow the data of each axis independently, and the
    vertical axis is scaled against the horizontal one by the ratio of the
    spans, so the frame stays square at any figure size without making one
    unit equally long on both axes.

    Parameters
    ----------
    figure : go.Figure
        Figure whose traces have ``x`` and ``y`` values.

    Returns
    -------
    go.Figure
        The same figure, updated in place.
    """
    x_range = _padded_range(
        [np.asarray(trace.x, dtype=np.float64) for trace in figure.data if trace.x is not None]
    )
    y_range = _padded_range(
        [np.asarray(trace.y, dtype=np.float64) for trace in figure.data if trace.y is not None]
    )
    zero_line = {"zeroline": True, "zerolinecolor": ZERO_LINE_COLOR, "zerolinewidth": 1}
    return figure.update_layout(
        xaxis={"range": x_range, "constrain": "domain", **zero_line},
        yaxis={
            "range": y_range,
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": (x_range[1] - x_range[0]) / (y_range[1] - y_range[0]),
            **zero_line,
        },
    )
