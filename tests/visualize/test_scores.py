"""Tests for the score scatter plot and the partial score trajectories."""

import numpy as np
import polars as pl

from flat_pca.visualize import create_partial_score_trajectories, create_score_scatter
from flat_pca.visualize.scores import MISSING_LABEL


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
    figure = create_score_scatter(_scores(), x="PC1", y="PC2", label="stem", color="lot")

    assert [trace.name for trace in figure.data] == ["L1", "L2", MISSING_LABEL]
    assert list(figure.data[0].customdata) == ["a", "d"]
    assert list(figure.data[0].x) == [1.0, 4.0]
    assert list(figure.data[0].y) == [-1.0, -4.0]
    assert figure.layout.xaxis.title.text == "PC1"
    assert figure.layout.yaxis.title.text == "PC2"
    assert figure.layout.legend.title.text == "lot"


def test_numeric_color_uses_a_color_scale() -> None:
    """A numeric column colors one trace continuously."""
    figure = create_score_scatter(
        _scores(), x="PC1", y="PC2", label="stem", color="yield_pct"
    )

    assert len(figure.data) == 1
    colors = np.asarray(figure.data[0].marker.color, dtype=np.float64)
    np.testing.assert_array_equal(colors[[0, 1, 3]], [90.0, 80.0, 70.0])
    assert np.isnan(colors[2])
    assert figure.data[0].marker.showscale
    assert list(figure.data[0].customdata) == ["a", "b", "c", "d"]


def test_without_color_draws_one_trace() -> None:
    """Without a color column, every point shares one trace."""
    figure = create_score_scatter(_scores(), x="PC2", y="PC1", label="stem")

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
