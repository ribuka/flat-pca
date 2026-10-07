"""Tests for the layout of component m vs component n plots."""

import plotly.graph_objects as go
import polars as pl
import pytest

from flat_pca.visualize import (
    create_loading_scatter,
    create_partial_score_trajectories,
    create_score_scatter,
)
from flat_pca.visualize.component_plane import (
    ZERO_LINE_COLOR,
    apply_component_plane_layout,
)


def _assert_square_frame(figure: go.Figure) -> None:
    """Assert that the frame is square and the zero lines are emphasized."""
    xaxis, yaxis = figure.layout.xaxis, figure.layout.yaxis
    x_span = xaxis.range[1] - xaxis.range[0]
    y_span = yaxis.range[1] - yaxis.range[0]
    assert yaxis.scaleanchor == "x"
    # Pixels per y unit are scaleratio times pixels per x unit, so the spans
    # take the same number of pixels.
    assert yaxis.scaleratio * y_span == pytest.approx(x_span)
    assert xaxis.constrain == yaxis.constrain == "domain"
    for axis in (xaxis, yaxis):
        assert axis.zeroline is True
        assert axis.zerolinecolor == ZERO_LINE_COLOR


def test_ranges_follow_each_axis_with_padding() -> None:
    """Each axis spans its own data over all traces, padded on both sides."""
    figure = go.Figure([go.Scatter(x=[0.0, 10.0], y=[1.0, 2.0]), go.Scatter(x=[5.0], y=[3.0])])

    apply_component_plane_layout(figure)

    assert tuple(figure.layout.xaxis.range) == pytest.approx((-0.5, 10.5))
    assert tuple(figure.layout.yaxis.range) == pytest.approx((0.9, 3.1))
    _assert_square_frame(figure)


def test_degenerate_values_get_a_finite_range() -> None:
    """Equal values span one unit, and no finite value spans -1 to 1."""
    figure = go.Figure(go.Scatter(x=[2.0, 2.0, float("nan")], y=[float("nan")]))

    apply_component_plane_layout(figure)

    assert tuple(figure.layout.xaxis.range) == pytest.approx((1.5, 2.5))
    assert tuple(figure.layout.yaxis.range) == pytest.approx((-1.0, 1.0))
    _assert_square_frame(figure)


def test_component_plots_use_the_square_frame() -> None:
    """Score, loading, and trajectory plots all draw a square frame."""
    loadings = pl.DataFrame({"wavelength": [400.0, 410.0], "PC1": [0.5, -0.5], "PC2": [0.1, 0.2]})
    figures = [
        create_score_scatter(
            [1.0, 2.0], [-30.0, 40.0], labels=["a", "b"], x_name="PC1", y_name="PC2"
        ),
        create_loading_scatter(loadings, x="PC1", y="PC2"),
        create_partial_score_trajectories(
            {"a": ([0.0, 1.0], [0.0, 100.0], ["p0", "p1"])}, x_name="PC1", y_name="PC2"
        ),
    ]

    for figure in figures:
        _assert_square_frame(figure)
