"""Tests for the score scatter plot and the partial score trajectories."""

import numpy as np
import polars as pl

from flat_pca.visualize import create_partial_score_trajectories, create_score_scatter
from flat_pca.visualize.marker_color import MISSING_LABEL


def _scores() -> pl.DataFrame:
    """Return the scores of four files with categorical and numeric metadata."""
    return pl.DataFrame(
        {
            "stem": ["a", "b", "c", "d"],
            "lot": ["L1", "L2", None, "L1"],
            "yield_pct": [90.0, 80.0, None, 70.0],
            "PC1": [1.0, 2.0, 3.0, 4.0],
            "PC2": [-1.0, -2.0, -3.0, -4.0],
        }
    )


def test_categorical_color_draws_one_trace_per_value() -> None:
    """Each value, including a missing one, becomes a named trace."""
    scores = _scores()
    figure = create_score_scatter(
        scores["PC1"],
        scores["PC2"],
        labels=scores["stem"],
        x_name="PC1",
        y_name="PC2",
        color=scores["lot"],
    )

    assert [trace.name for trace in figure.data] == ["L1", "L2", MISSING_LABEL]
    assert list(figure.data[0].customdata) == ["a", "d"]
    assert list(figure.data[0].x) == [1.0, 4.0]
    assert list(figure.data[0].y) == [-1.0, -4.0]
    assert figure.layout.xaxis.title.text == "PC1"
    assert figure.layout.yaxis.title.text == "PC2"
    assert figure.layout.legend.title.text == "lot"


def test_numeric_color_uses_a_color_scale() -> None:
    """A numeric column colors one trace continuously."""
    scores = _scores()
    figure = create_score_scatter(
        scores["PC1"],
        scores["PC2"],
        labels=scores["stem"],
        x_name="PC1",
        y_name="PC2",
        color=scores["yield_pct"],
    )

    assert len(figure.data) == 1
    colors = np.asarray(figure.data[0].marker.color, dtype=np.float64)
    np.testing.assert_array_equal(colors[[0, 1, 3]], [90.0, 80.0, 70.0])
    assert np.isnan(colors[2])
    assert figure.data[0].marker.showscale
    assert list(figure.data[0].customdata) == ["a", "b", "c", "d"]


def test_without_color_draws_one_trace() -> None:
    """Without a color column, every point shares one trace."""
    scores = _scores()
    figure = create_score_scatter(
        scores["PC2"], scores["PC1"], labels=scores["stem"], x_name="PC2", y_name="PC1"
    )

    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [-1.0, -2.0, -3.0, -4.0]
    assert figure.data[0].marker.color is None


def test_trajectories_mark_their_end_points() -> None:
    """Each trajectory is a named line whose last point is drawn larger."""
    figure = create_partial_score_trajectories(
        {
            "a": ([0.0, 1.0, 3.0], [0.0, -1.0, 2.0], ["p1", "p2", "p3"]),
            "b": (np.array([1.0]), np.array([2.0]), ["p1"]),
        },
        x_name="PC1",
        y_name="PC3",
    )

    assert [trace.name for trace in figure.data] == ["a", "b"]
    assert list(figure.data[0].x) == [0.0, 1.0, 3.0]
    assert list(figure.data[0].text) == ["p1", "p2", "p3"]
    sizes = list(figure.data[0].marker.size)
    assert sizes[-1] > sizes[0]
    assert figure.layout.xaxis.title.text == "PC1"
    assert figure.layout.yaxis.title.text == "PC3"


def _four_trajectories() -> dict[str, tuple[list[float], list[float], list[str]]]:
    """Return two-point trajectories of the files ``a`` to ``d``."""
    return {
        stem: ([0.0, float(index)], [0.0, -float(index)], ["p1", "p2"])
        for index, stem in enumerate(["a", "b", "c", "d"])
    }


def test_trajectories_colored_by_stem_get_one_color_each() -> None:
    """Colored by the file name, every trajectory has its own color and legend entry."""
    scores = _scores()
    figure = create_partial_score_trajectories(
        _four_trajectories(), x_name="PC1", y_name="PC2", color=scores["stem"]
    )

    assert [trace.name for trace in figure.data] == ["a", "b", "c", "d"]
    assert all(trace.showlegend for trace in figure.data)
    colors = [trace.line.color for trace in figure.data]
    assert len(set(colors)) == 4
    assert [trace.marker.color for trace in figure.data] == colors
    assert figure.layout.legend.title.text == "stem"


def test_trajectories_colored_by_a_category_share_the_value_color() -> None:
    """Trajectories of one value share a color and a single legend entry."""
    scores = _scores()
    figure = create_partial_score_trajectories(
        _four_trajectories(), x_name="PC1", y_name="PC2", color=scores["lot"]
    )

    assert [trace.name for trace in figure.data] == ["L1", "L2", MISSING_LABEL, "L1"]
    assert [trace.showlegend for trace in figure.data] == [True, True, True, False]
    assert [trace.legendgroup for trace in figure.data] == ["L1", "L2", MISSING_LABEL, "L1"]
    colors = [trace.line.color for trace in figure.data]
    assert colors[0] == colors[3]
    assert len(set(colors)) == 3
    # The hover text still names the file.
    assert figure.data[3].hovertemplate.startswith("d<br>")


def test_trajectories_colored_by_a_number_share_one_scale() -> None:
    """A numeric value colors lines and points on one scale with one color bar."""
    scores = _scores()
    figure = create_partial_score_trajectories(
        _four_trajectories(), x_name="PC1", y_name="PC2", color=scores["yield_pct"]
    )

    assert not any(trace.showlegend for trace in figure.data)
    assert [bool(trace.marker.showscale) for trace in figure.data] == [True, False, False, False]
    assert figure.data[0].marker.colorbar.title.text == "yield_pct"
    assert list(figure.data[0].marker.color) == [90.0, 90.0]
    assert (figure.data[0].marker.cmin, figure.data[0].marker.cmax) == (70.0, 90.0)
    lines = [trace.line.color for trace in figure.data]
    # The ends of Viridis, and grey for the missing value.
    assert lines[0] == "rgb(253, 231, 37)"
    assert lines[3] == "rgb(68, 1, 84)"
    assert lines[2] == "#9e9e9e"


def test_trajectories_without_color_share_one_color() -> None:
    """Without a color value, every trajectory has the same color and no legend."""
    figure = create_partial_score_trajectories(
        _four_trajectories(), x_name="PC1", y_name="PC2"
    )

    assert len({trace.line.color for trace in figure.data}) == 1
    assert not any(trace.showlegend for trace in figure.data)
    assert [trace.name for trace in figure.data] == ["a", "b", "c", "d"]


def test_trajectories_colored_by_numbers_near_the_float_limit() -> None:
    """A finite range wider than the float limit still colors the ends of the scale."""
    figure = create_partial_score_trajectories(
        {name: _four_trajectories()[name] for name in ("a", "b")},
        x_name="PC1",
        y_name="PC2",
        color=pl.Series("value", [-1e308, 1e308]),
    )

    assert [trace.line.color for trace in figure.data] == [
        "rgb(68, 1, 84)",
        "rgb(253, 231, 37)",
    ]


def test_trajectories_with_a_missing_number_are_grey() -> None:
    """A trajectory without a value is grey, and the first one with a value draws the bar."""
    figure = create_partial_score_trajectories(
        _four_trajectories(),
        x_name="PC1",
        y_name="PC2",
        color=pl.Series("yield_pct", [None, 1.0, None, 2.0]),
    )

    for missing in (figure.data[0], figure.data[2]):
        assert missing.line.color == "#9e9e9e"
        assert missing.marker.color == "#9e9e9e"
        assert not missing.marker.showscale
    assert [bool(trace.marker.showscale) for trace in figure.data] == [False, True, False, False]
    assert figure.data[1].marker.colorbar.title.text == "yield_pct"
    assert (figure.data[3].marker.cmin, figure.data[3].marker.cmax) == (1.0, 2.0)


def test_trajectories_without_any_number_are_all_grey() -> None:
    """A numeric value missing for every trajectory draws them grey without a color bar."""
    figure = create_partial_score_trajectories(
        _four_trajectories(),
        x_name="PC1",
        y_name="PC2",
        color=pl.Series("yield_pct", [None] * 4, dtype=pl.Float64),
    )

    assert {trace.marker.color for trace in figure.data} == {"#9e9e9e"}
    assert {trace.line.color for trace in figure.data} == {"#9e9e9e"}
    assert not any(trace.marker.showscale for trace in figure.data)
