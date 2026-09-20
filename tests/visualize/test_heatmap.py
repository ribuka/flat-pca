"""Tests for spectral plot helpers."""

import numpy as np
import polars as pl

from flat_pca.visualize import create_heatmap
from flat_pca.visualize.heatmap import _symmetric_color_range


def test_symmetric_color_range_uses_abs_quantile() -> None:
    """Derive a zero-centered range from the quantile of abs(z)."""
    z = np.array([[-10.0, 2.0], [4.0, 8.0]])

    assert _symmetric_color_range(z, quantile=1.0) == (-10.0, 10.0)


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

    figure = create_heatmap(spectra)

    heatmap = figure.data[0]
    assert list(heatmap.x) == [650.0, 651.0, 652.0]
    assert list(heatmap.y) == [0.0, 1.0]
    assert heatmap.z.tolist() == [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]
    assert figure.layout.coloraxis.colorscale is not None
    assert figure.layout.coloraxis.cmin is None
    assert figure.layout.coloraxis.cmax is None


def test_create_heatmap_auto_applies_diverging_scale_for_coefficient() -> None:
    """z_name='coefficient' should get a symmetric RdBu_r color range."""
    spectra = pl.DataFrame(
        {
            "wavelength": [650.0, 650.0, 651.0, 651.0],
            "Time": [0.0, 1.0, 0.0, 1.0],
            "coefficient": [-10.0, 2.0, 4.0, 8.0],
        }
    )

    figure = create_heatmap(
        spectra,
        z_name="coefficient",
        symmetric_range_quantile=1.0,
    )

    assert figure.layout.coloraxis.cmin == -10.0
    assert figure.layout.coloraxis.cmax == 10.0


def test_create_heatmap_respects_explicit_range_color_for_coefficient() -> None:
    """An explicit range_color overrides the automatic symmetric range."""
    spectra = pl.DataFrame(
        {
            "wavelength": [650.0, 650.0, 651.0, 651.0],
            "Time": [0.0, 1.0, 0.0, 1.0],
            "coefficient": [-10.0, 2.0, 4.0, 8.0],
        }
    )

    figure = create_heatmap(
        spectra,
        z_name="coefficient",
        range_color=(-1.0, 1.0),
    )

    assert figure.layout.coloraxis.cmin == -1.0
    assert figure.layout.coloraxis.cmax == 1.0
