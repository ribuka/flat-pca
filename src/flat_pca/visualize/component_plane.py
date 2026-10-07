"""Layout shared by plots of component m against component n."""

import plotly.graph_objects as go

ZERO_LINE_COLOR = "#888"
FRAME_SIZE = 400
MARGIN = {"l": 80, "r": 160, "t": 60, "b": 70}


def apply_component_plane_layout(figure: go.Figure) -> go.Figure:
    """Draw a component plane in a square frame with emphasized zero lines.

    The figure size and margins are fixed so that the frame, the area
    enclosed by the axes, is ``FRAME_SIZE`` pixels on each side whatever
    the window size or the axis ranges. The margins do not grow for the
    legend or the color bar, which are drawn in the right margin. The axis
    ranges stay automatic on each axis, so one unit need not be equally
    long on both axes.

    Parameters
    ----------
    figure : go.Figure
        Figure to update.

    Returns
    -------
    go.Figure
        The same figure, updated in place.
    """
    zero_line = {"zeroline": True, "zerolinecolor": ZERO_LINE_COLOR, "zerolinewidth": 1}
    return figure.update_layout(
        autosize=False,
        width=MARGIN["l"] + FRAME_SIZE + MARGIN["r"],
        height=MARGIN["t"] + FRAME_SIZE + MARGIN["b"],
        margin={**MARGIN, "autoexpand": False},
        xaxis=zero_line,
        yaxis=zero_line,
    )
