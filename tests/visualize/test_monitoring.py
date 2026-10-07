"""Tests for the T² and Q control charts and scatter plot."""

from flat_pca.visualize import create_control_chart, create_t2_q_scatter
from flat_pca.visualize.monitoring import EXCEEDS_UCL_NAME, WITHIN_UCL_NAME


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
