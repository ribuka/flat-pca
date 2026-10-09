"""Coloring of scatter markers by a metadata value."""

import numpy as np
import plotly.colors
import polars as pl

MISSING_LABEL = "(missing)"
CONTINUOUS_COLOR_SCALE = "Viridis"
# Trajectories without a numeric color value are drawn grey.
MISSING_COLOR = "#9e9e9e"
# Scatter points without a numeric color value are drawn faint, so the
# points with a value stand out.
MISSING_MARKER = {"color": "lightgray", "opacity": 0.5}
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
        ``showscale`` of a Plotly marker. ``color`` must have a finite value
        to span the range.
    """
    values = color.cast(pl.Float64).to_numpy()
    finite = values[np.isfinite(values)]
    return {
        "color": values if rows is None else values[rows],
        "colorscale": CONTINUOUS_COLOR_SCALE,
        "cmin": float(finite.min()),
        "cmax": float(finite.max()),
        "colorbar": {"title": {"text": color.name}},
        "showscale": showscale,
    }


def continuous_marker_groups(
    color: pl.Series, rows: np.ndarray | None = None
) -> list[tuple[np.ndarray, dict[str, object]]]:
    """Split points colored by a numeric value into missing and valued ones.

    Plotly draws later traces in front, so drawing one trace per group in
    the order returned puts the points with a value in front of the faint
    points without one.

    Parameters
    ----------
    color : pl.Series
        Numeric color value of every point, titled by the series name.
    rows : np.ndarray | None, optional
        Ascending row positions of the points drawn; all rows by default.
        The color range spans all of ``color``, so figures drawing different
        rows of the same series color them alike.

    Returns
    -------
    list[tuple[np.ndarray, dict[str, object]]]
        ``(rows, marker)`` of the drawn points without a finite value, marked
        with ``MISSING_MARKER``, then of those with one, marked by
        ``continuous_marker`` with its color bar. A group without points is
        left out.
    """
    values = color.cast(pl.Float64).to_numpy()
    drawn = np.arange(values.size) if rows is None else np.asarray(rows, dtype=np.intp)
    finite = np.isfinite(values[drawn])
    groups: list[tuple[np.ndarray, dict[str, object]]] = []
    if not finite.all():
        groups.append((drawn[~finite], dict(MISSING_MARKER)))
    if finite.any():
        groups.append((drawn[finite], continuous_marker(color, drawn[finite])))
    return groups
