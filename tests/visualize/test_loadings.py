"""Tests for the loading scatter plot."""

import polars as pl

from flat_pca.visualize import create_loading_scatter


def test_loading_scatter_draws_one_point_per_wavelength() -> None:
    """Points are placed by the two loadings and colored by wavelength."""
    loadings = pl.DataFrame(
        {"wavelength": [400.0, 410.0], "PC1": [0.5, -0.5], "PC2": [0.1, 0.2]}
    )

    figure = create_loading_scatter(loadings, x="PC1", y="PC2")

    assert len(figure.data) == 1
    trace = figure.data[0]
    assert list(trace.x) == [0.5, -0.5]
    assert list(trace.y) == [0.1, 0.2]
    assert list(trace.marker.color) == [400.0, 410.0]
    assert list(trace.customdata) == [400.0, 410.0]
    assert figure.layout.xaxis.title.text == "PC1"
