"""Tests for the T² and Q control charts and scatter plot."""

import numpy as np
import polars as pl

from flat_pca.visualize import create_control_chart, create_t2_q_scatter
from flat_pca.visualize.marker_color import MISSING_COLOR
from flat_pca.visualize.monitoring import (
    EXCEEDS_UCL_COLOR,
    EXCEEDS_UCL_NAME,
    WITHIN_UCL_NAME,
)


def test_control_chart_highlights_points_above_the_limit() -> None:
    """Points above the UCL form their own trace at their file positions."""
    figure = create_control_chart(
        [1.0, 5.0, 2.0, 7.0],
        ucl=4.0,
        labels=["a", "b", "c", "d"],
        y_name="Q",
        order_values=["2024-01", "2024-02", "2024-03", "2024-04"],
        order_name="date",
    )

    within, exceeds = figure.data
    assert (within.name, exceeds.name) == (WITHIN_UCL_NAME, EXCEEDS_UCL_NAME)
    assert list(within.x) == [1, 3]
    assert list(within.customdata) == ["a", "c"]
    assert list(exceeds.x) == [2, 4]
    assert list(exceeds.y) == [5.0, 7.0]
    assert list(exceeds.text) == ["2024-02", "2024-04"]
    assert "date=%{text}" in exceeds.hovertemplate
    assert figure.layout.shapes[0].y0 == 4.0
    assert figure.layout.shapes[0].y1 == 4.0
    assert figure.layout.xaxis.title.text == "file order (date)"
    assert figure.layout.yaxis.title.text == "Q"


def test_control_chart_without_an_order_column() -> None:
    """Without an ordering column the hover text omits it."""
    figure = create_control_chart([1.0], ucl=4.0, labels=["a"], y_name="T²")

    assert figure.data[0].text is None
    assert "%{text}" not in figure.data[0].hovertemplate
    assert figure.layout.xaxis.title.text == "file order"
    assert len(figure.data[1].x) == 0


def test_scatter_highlights_points_above_either_limit() -> None:
    """A point above the T² or the Q limit is highlighted."""
    figure = create_t2_q_scatter(
        [1.0, 9.0, 1.0, 2.0],
        [1.0, 1.0, 9.0, 2.0],
        t2_ucl=5.0,
        q_ucl=5.0,
        labels=["a", "b", "c", "d"],
    )

    within, exceeds = figure.data
    assert list(within.customdata) == ["a", "d"]
    assert list(exceeds.customdata) == ["b", "c"]
    vertical, horizontal = figure.layout.shapes
    assert (vertical.x0, vertical.x1) == (5.0, 5.0)
    assert (horizontal.y0, horizontal.y1) == (5.0, 5.0)
    assert figure.layout.xaxis.title.text == "T²"
    assert figure.layout.yaxis.title.text == "Q"


def test_categorical_color_splits_each_value_at_the_limit() -> None:
    """Each value has a within and an above-limit trace in one legend group and color."""
    figure = create_control_chart(
        [1.0, 5.0, 2.0, 7.0],
        ucl=4.0,
        labels=["a", "b", "c", "d"],
        y_name="Q",
        color=pl.Series("lot", ["L1", "L1", "L2", None]),
    )

    names = [trace.name for trace in figure.data]
    assert names == ["L1", "L1 (exceeds UCL)", "L2", "L2 (exceeds UCL)", "(missing)", "(missing) (exceeds UCL)"]
    assert [list(trace.customdata) for trace in figure.data] == [["a"], ["b"], ["c"], [], [], ["d"]]
    assert [trace.legendgroup for trace in figure.data[:2]] == ["L1", "L1"]
    assert figure.data[0].marker.color == figure.data[1].marker.color
    assert figure.data[1].marker.symbol == "diamond"
    assert figure.data[1].marker.line.color == EXCEEDS_UCL_COLOR
    # A value without points above the limit lists no such entry in the legend.
    assert figure.data[3].showlegend is False
    assert figure.layout.legend.title.text == "lot"


def test_numeric_color_shares_one_scale_across_the_limit() -> None:
    """Both traces of a numeric color use the full value range of one scale."""
    figure = create_t2_q_scatter(
        [1.0, 9.0, 2.0],
        [1.0, 1.0, 2.0],
        t2_ucl=5.0,
        q_ucl=5.0,
        labels=["a", "b", "c"],
        color=pl.Series("yield_pct", [90.0, 70.0, None]),
    )

    within, exceeds = figure.data
    assert (within.name, exceeds.name) == (WITHIN_UCL_NAME, EXCEEDS_UCL_NAME)
    colors = np.asarray(within.marker.color, dtype=np.float64)
    assert colors[0] == 90.0 and np.isnan(colors[1])
    assert list(exceeds.marker.color) == [70.0]
    for trace in (within, exceeds):
        assert (trace.marker.cmin, trace.marker.cmax) == (70.0, 90.0)
    assert within.marker.showscale is True
    assert exceeds.marker.showscale is False
    assert exceeds.marker.line.color == EXCEEDS_UCL_COLOR
    assert figure.layout.legend.orientation == "h"


def test_numeric_color_keeps_one_scale_when_every_point_exceeds() -> None:
    """With no point within the limit, the above-limit trace draws the only color bar."""
    figure = create_control_chart(
        [5.0, 7.0], ucl=4.0, labels=["a", "b"], y_name="Q", color=pl.Series("yield_pct", [90.0, 70.0])
    )

    within, exceeds = figure.data
    assert within.marker.showscale is False
    assert exceeds.marker.showscale is True


def test_numeric_color_without_values_draws_no_scale() -> None:
    """A numeric column missing everywhere colors the points grey without a color bar."""
    figure = create_t2_q_scatter(
        [1.0, 9.0],
        [1.0, 1.0],
        t2_ucl=5.0,
        q_ucl=5.0,
        labels=["a", "b"],
        color=pl.Series("yield_pct", [None, None], dtype=pl.Float64),
    )

    for trace in figure.data:
        assert trace.marker.color == MISSING_COLOR
        assert trace.marker.showscale is False
        assert trace.marker.colorbar.title.text is None


def test_numeric_color_scale_goes_on_the_trace_with_values() -> None:
    """When only points above the limit have color values, their trace draws the color bar."""
    figure = create_control_chart(
        [1.0, 7.0, 2.0],
        ucl=4.0,
        labels=["a", "b", "c"],
        y_name="Q",
        color=pl.Series("yield_pct", [None, 70.0, None], dtype=pl.Float64),
    )

    within, exceeds = figure.data
    assert within.marker.showscale is False
    assert exceeds.marker.showscale is True
