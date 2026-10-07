"""Tests for the layout of component m vs component n plots."""

import plotly.graph_objects as go
import polars as pl

from flat_pca.visualize import (
    create_loading_scatter,
    create_partial_score_trajectories,
    create_score_scatter,
)
from flat_pca.visualize.component_plane import FRAME_SIZE, ZERO_LINE_COLOR


def _assert_square_frame(figure: go.Figure) -> None:
    """Assert that the frame is square and the zero lines are emphasized."""
    layout = figure.layout
    assert layout.autosize is False
    assert layout.margin.autoexpand is False
    assert layout.width - layout.margin.l - layout.margin.r == FRAME_SIZE
    assert layout.height - layout.margin.t - layout.margin.b == FRAME_SIZE
    for axis in (layout.xaxis, layout.yaxis):
        assert axis.zeroline is True
        assert axis.zerolinecolor == ZERO_LINE_COLOR
        assert axis.range is None
        assert axis.scaleanchor is None


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
