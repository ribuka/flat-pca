"""Coloring of scatter markers by a metadata value."""

import numpy as np
import plotly.colors
import polars as pl

MISSING_LABEL = "(missing)"
CONTINUOUS_COLOR_SCALE = "Viridis"
# Points without a numeric color value are drawn grey.
MISSING_COLOR = "#9e9e9e"
CATEGORY_COLORS = plotly.colors.qualitative.Plotly


def color_groups(color: pl.Series) -> dict[str, np.ndarray]:
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


def category_color(index: int) -> str:
    """Return the color of the ``index``-th categorical value.

    Parameters
    ----------
    index : int
        Zero-based position of the value among the groups.

    Returns
    -------
    str
        Color of Plotly's default qualitative palette, repeating after its end.
    """
    return CATEGORY_COLORS[index % len(CATEGORY_COLORS)]


def continuous_marker(
    color: pl.Series, rows: np.ndarray | None = None, *, showscale: bool = True
) -> dict[str, object]:
    """Return marker settings coloring points by a numeric value.

    Parameters
    ----------
    color : pl.Series
        Numeric color value of every point, titled by the series name.
    rows : np.ndarray | None, optional
        Row positions of the points of one trace. The color range spans all
        of ``color``, so several traces share one scale. All rows by default.
    showscale : bool, default True
        Whether this trace draws the color bar. Plotly.js draws one for every
        trace with a ``colorbar`` unless told otherwise, so traces sharing a
        scale set it on exactly one of them.

    Returns
    -------
    dict[str, object]
        ``color``, ``colorscale``, ``cmin``, ``cmax``, ``colorbar``, and
        ``showscale`` of a Plotly marker. Without any finite value there is no
        range to show, so every point gets ``MISSING_COLOR`` and no color bar.
    """
    values = color.cast(pl.Float64).to_numpy()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"color": MISSING_COLOR, "showscale": False}
    return {
        "color": values if rows is None else values[rows],
        "colorscale": CONTINUOUS_COLOR_SCALE,
        "cmin": float(finite.min()),
        "cmax": float(finite.max()),
        "colorbar": {"title": {"text": color.name}},
        "showscale": showscale,
    }
