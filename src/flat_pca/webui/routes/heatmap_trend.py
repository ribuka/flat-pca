"""Heatmap with trends shared by the exploration and model screens."""

from __future__ import annotations

import json

from fastapi.responses import Response

from flat_pca.visualize import create_heatmap, create_trend

from ..services.explore import explore_trends
from ..services.heatmap_binning import bin_step_times
from ..services.spectral_matrix import SpectralMatrix

# Fixed top margin of the heatmap, leaving room for the title and the
# selection marker above the plot area.
HEATMAP_MARGIN_TOP = 72
# Gap between the plot area and the StepTime tick labels, where the selection
# marker on the left is drawn.
MARKER_GAP = 16


def script_json(text: str) -> str:
    """Make JSON text safe to embed in a ``<script>`` element.

    Parameters
    ----------
    text : str
        JSON text.

    Returns
    -------
    str
        ``text`` with ``</`` escaped so it cannot close the element.
    """
    return text.replace("</", "<\\/")


def heatmap_context(
    matrices: dict[str, SpectralMatrix], value_name: str, max_cells: int, trend_url: str
) -> dict[str, object]:
    """Build the template context of ``partials/heatmap_trend.html``.

    Parameters
    ----------
    matrices : dict[str, SpectralMatrix]
        Unbinned matrices keyed by trace label; the first is the heatmap.
    value_name : str
        Name of the values, shown on the color bar.
    max_cells : int
        Cell count above which the heatmap is binned along ``StepTime``.
    trend_url : str
        URL of the trend requests, with a query string to which the point
        is appended.

    Returns
    -------
    dict[str, object]
        Heatmap label, binning, figure and axes JSON, axis sizes, and the
        trend URL.
    """
    label, matrix = next(iter(matrices.items()))
    binned = bin_step_times(matrix, max_cells)
    figure = create_heatmap(
        z=binned.matrix.values,
        x=binned.matrix.wavelengths,
        y=binned.matrix.step_times,
        y_name="StepTime",
        z_name=value_name,
    )
    figure.update_layout(title=label, margin={"t": HEATMAP_MARGIN_TOP})
    figure.update_yaxes(ticks="outside", ticklen=MARKER_GAP, tickcolor="rgba(0, 0, 0, 0)")
    return {
        "heatmap_label": label,
        "binned": binned,
        "max_cells": max_cells,
        "figure_json": script_json(figure.to_json()),
        "axes_json": script_json(
            json.dumps(
                {
                    "wavelengths": matrix.wavelengths.tolist(),
                    "step_times": matrix.step_times.tolist(),
                }
            )
        ),
        "n_wavelengths": matrix.wavelengths.size,
        "n_step_times": matrix.step_times.size,
        "trend_url": trend_url,
    }


def trend_response(
    matrices: dict[str, SpectralMatrix], value_name: str, wavelength: float, step_time: float
) -> Response:
    """Return the trend figures at one point, cut from the unbinned values.

    Parameters
    ----------
    matrices : dict[str, SpectralMatrix]
        Unbinned matrices keyed by trace label.
    value_name : str
        Name of the values, shown on the vertical axes.
    wavelength : float
        Requested wavelength; each trace uses its nearest one.
    step_time : float
        Requested ``StepTime``; each trace uses its nearest one.

    Returns
    -------
    Response
        JSON with ``wavelength`` and ``step_time`` (the heatmap trace's grid
        point), ``by_step_time`` (figure of the values over ``StepTime`` at
        the wavelength), and ``by_wavelength`` (figure of the values over
        wavelength at the ``StepTime``).
    """
    by_time, by_wavelength = explore_trends(matrices, wavelength, step_time)
    first_time = next(iter(by_time.values()))
    first_wavelength = next(iter(by_wavelength.values()))
    time_figure = create_trend(
        {label: (line.x, line.y) for label, line in by_time.items()},
        x_name="StepTime",
        y_name=value_name,
        title=f"wavelength = {first_time.at:g}",
    )
    wavelength_figure = create_trend(
        {label: (line.x, line.y) for label, line in by_wavelength.items()},
        x_name="wavelength",
        y_name=value_name,
        title=f"StepTime = {first_wavelength.at:g}",
    )
    body = (
        f'{{"wavelength": {json.dumps(first_time.at)}, '
        f'"step_time": {json.dumps(first_wavelength.at)}, '
        f'"by_step_time": {time_figure.to_json()}, '
        f'"by_wavelength": {wavelength_figure.to_json()}}}'
    )
    return Response(content=body, media_type="application/json")
