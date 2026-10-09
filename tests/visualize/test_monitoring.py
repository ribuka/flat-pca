"""Tests for the T² and Q control charts and scatter plot."""

import json

import numpy as np
import polars as pl

from flat_pca.visualize import create_control_chart, create_t2_q_scatter
from flat_pca.visualize.marker_color import MISSING_COLOR
from flat_pca.visualize.monitoring import MARKER


def test_control_chart_draws_every_point_alike_in_file_order() -> None:
    """Points above the UCL share the trace and marker of the others."""
    figure = create_control_chart(
        [1.0, 5.0, 2.0, 7.0],
        ucl=4.0,
        labels=["a", "b", "c", "d"],
        y_name="Q",
        x_text=["2024-01", "2024-02", "2024-03", "2024-04"],
        x_name="date",
    )

    (trace,) = figure.data
    assert list(trace.x) == [1, 2, 3, 4]
    assert list(trace.y) == [1.0, 5.0, 2.0, 7.0]
    assert list(trace.customdata) == ["a", "b", "c", "d"]
    assert list(trace.text) == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert trace.marker.size == MARKER["size"]
    assert trace.marker.symbol is None
    assert "date=%{text}" in trace.hovertemplate
    assert figure.layout.shapes[0].y0 == 4.0
    assert figure.layout.shapes[0].y1 == 4.0
    assert figure.layout.xaxis.title.text == "file order (date)"
    assert figure.layout.yaxis.title.text == "Q"


def test_control_chart_without_an_x_axis_column() -> None:
    """Without a horizontal-axis column the hover text omits it."""
    figure = create_control_chart([1.0], ucl=4.0, labels=["a"], y_name="T²")

    assert figure.data[0].text is None
    assert "%{text}" not in figure.data[0].hovertemplate
    assert figure.layout.xaxis.title.text == "file order"


def test_control_chart_on_a_numeric_axis() -> None:
    """Numeric ``x`` values place the points and title the axis by the column."""
    figure = create_control_chart(
        [1.0, 5.0, 2.0],
        ucl=4.0,
        labels=["a", "b", "c"],
        y_name="Q",
        x=pl.Series("yield_pct", [90, 70, 85]),
        x_text=["90", "70", "85"],
        x_name="yield_pct",
    )

    (trace,) = figure.data
    assert list(trace.x) == [90.0, 70.0, 85.0]
    assert "yield_pct=%{text}" in trace.hovertemplate
    assert figure.layout.xaxis.title.text == "yield_pct"


def test_control_chart_on_a_date_axis() -> None:
    """Datetime ``x`` values are sent as dates, which Plotly draws on a date axis."""
    figure = create_control_chart(
        [1.0, 5.0],
        ucl=4.0,
        labels=["a", "b"],
        y_name="T²",
        x=pl.Series("date", ["2024-01-02T00:00:00", "2024-01-01T12:00:00"]).str.to_datetime(),
        x_name="date",
    )

    sent = json.loads(figure.to_json())["data"][0]["x"]
    assert sent == ["2024-01-02T00:00:00", "2024-01-01T12:00:00"]
    assert figure.layout.xaxis.title.text == "date"


def test_scatter_draws_every_point_alike() -> None:
    """Points above either limit share the trace of the others."""
    figure = create_t2_q_scatter(
        [1.0, 9.0, 1.0, 2.0],
        [1.0, 1.0, 9.0, 2.0],
        t2_ucl=5.0,
        q_ucl=5.0,
        labels=["a", "b", "c", "d"],
    )

    (trace,) = figure.data
    assert list(trace.customdata) == ["a", "b", "c", "d"]
    assert trace.marker.size == MARKER["size"]
    vertical, horizontal = figure.layout.shapes
    assert (vertical.x0, vertical.x1) == (5.0, 5.0)
    assert (horizontal.y0, horizontal.y1) == (5.0, 5.0)
    assert figure.layout.xaxis.title.text == "T²"
    assert figure.layout.yaxis.title.text == "Q"


def test_categorical_color_draws_one_trace_per_value() -> None:
    """Each value, missing ones included, has one trace and color in the legend."""
    figure = create_control_chart(
        [1.0, 5.0, 2.0, 7.0],
        ucl=4.0,
        labels=["a", "b", "c", "d"],
        y_name="Q",
        color=pl.Series("lot", ["L1", "L1", "L2", None]),
    )

    assert [trace.name for trace in figure.data] == ["L1", "L2", "(missing)"]
    assert [list(trace.customdata) for trace in figure.data] == [["a", "b"], ["c"], ["d"]]
    assert len({trace.marker.color for trace in figure.data}) == 3
    assert {trace.marker.symbol for trace in figure.data} == {None}
    assert figure.layout.legend.title.text == "lot"


def test_numeric_color_draws_one_scale() -> None:
    """A numeric color draws one trace with the full value range and its color bar."""
    figure = create_t2_q_scatter(
        [1.0, 9.0, 2.0],
        [1.0, 1.0, 2.0],
        t2_ucl=5.0,
        q_ucl=5.0,
        labels=["a", "b", "c"],
        color=pl.Series("yield_pct", [90.0, 70.0, None]),
    )

    (trace,) = figure.data
    colors = np.asarray(trace.marker.color, dtype=np.float64)
    assert colors[:2].tolist() == [90.0, 70.0] and np.isnan(colors[2])
    assert (trace.marker.cmin, trace.marker.cmax) == (70.0, 90.0)
    assert trace.marker.showscale is True
    assert trace.marker.colorbar.title.text == "yield_pct"
    assert figure.layout.legend.title.text is None


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

    (trace,) = figure.data
    assert trace.marker.color == MISSING_COLOR
    assert trace.marker.showscale is False
    assert trace.marker.colorbar.title.text is None
