"""Tests for spectral plot helpers."""

import polars as pl

from spca.visualize import create_spectra_heatmap


def test_create_spectra_heatmap_ignores_metadata_columns() -> None:
    """Use every non-metadata column as a numeric wavelength."""
    spectra = pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Step": [1, 2],
            "Sequence": ["a", "b"],
            "652.0nm": [50.0, 60.0],
            "650.0nm": [10.0, 20.0],
            "651.0nm": [30.0, 40.0],
        }
    )

    figure = create_spectra_heatmap(spectra)

    heatmap = figure.data[0]
    assert list(heatmap.x) == [650.0, 651.0, 652.0]
    assert list(heatmap.y) == [0.0, 1.0]
    assert heatmap.z.tolist() == [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]
