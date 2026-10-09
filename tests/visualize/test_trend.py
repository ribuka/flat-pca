"""Tests for the trend line plot."""

import numpy as np

from flat_pca.visualize import create_trend
from flat_pca.visualize.trend import TREND_FRAME_HEIGHT, TREND_MARGIN, trend_height


def test_create_trend_draws_one_trace_per_label() -> None:
    """Each labeled series becomes a named line in mapping order."""
    figure = create_trend(
        {
            "run-1": ([0.0, 1.0], [2.0, np.nan]),
            "run-2": (np.array([0.0, 2.0]), np.array([3.0, 4.0])),
        },
        x_name="StepTime",
        title="wavelength = 400",
    )

    assert [trace.name for trace in figure.data] == ["run-1", "run-2"]
    assert list(figure.data[1].x) == [0.0, 2.0]
    assert np.isnan(figure.data[0].y[1])
    assert figure.layout.xaxis.title.text == "StepTime"
    assert figure.layout.yaxis.title.text == "intensity"
    assert figure.layout.title.text == "wavelength = 400"


def test_trend_height_adds_the_margins_and_the_legend() -> None:
    """The figure height is the plot area plus the margins and the legend."""
    assert trend_height(300) == TREND_MARGIN["t"] + 300 + TREND_MARGIN["b"]
    assert trend_height(300, legend_height=45) == trend_height(300) + 45


def test_create_trend_puts_the_legend_above_the_plot_area() -> None:
    """A horizontal legend sits above the plot area, and the height leaves no room for it."""
    figure = create_trend({"run-1": ([0.0], [1.0])}, x_name="StepTime", frame_height=250)

    legend = figure.layout.legend
    assert legend.orientation == "h"
    assert (legend.y, legend.yanchor) == (1, "bottom")
    assert legend.maxheight == 1
    assert figure.layout.margin.t == TREND_MARGIN["t"]
    assert figure.layout.margin.b == TREND_MARGIN["b"]
    assert figure.layout.height == trend_height(250)
    assert figure.layout.autosize is True
    assert create_trend({}, x_name="StepTime").layout.height == trend_height(TREND_FRAME_HEIGHT)
