"""Tests for spectral plot helpers."""

import numpy as np
import polars as pl
import pytest

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


def test_create_heatmap_ignores_step_time_columns() -> None:
    """Exclude StepTime and ReverseStepTime from the wavelength columns."""
    spectra = pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Step": [1, 1],
            "Sequence": [0, 0],
            "StepTime": [0.0, 1.0],
            "ReverseStepTime": [1.0, 0.0],
            "651.0nm": [30.0, 40.0],
            "650.0nm": [10.0, 20.0],
        }
    )

    figure = create_heatmap(spectra)

    heatmap = figure.data[0]
    assert list(heatmap.x) == [650.0, 651.0]
    assert heatmap.z.tolist() == [[10.0, 30.0], [20.0, 40.0]]


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


def test_create_heatmap_from_matrix_matches_the_frame_path() -> None:
    """A matrix with its axes draws the same heatmap as the equivalent frame."""
    spectra = pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Step": [1, 1],
            "Sequence": [0, 0],
            "651.0nm": [30.0, 40.0],
            "650.0nm": [10.0, 20.0],
        }
    )

    from_frame = create_heatmap(spectra)
    from_matrix = create_heatmap(
        z=np.array([[10.0, 30.0], [20.0, 40.0]]), x=[650.0, 651.0], y=[0.0, 1.0]
    )

    assert from_matrix.to_plotly_json() == from_frame.to_plotly_json()


def test_create_heatmap_from_matrix_keeps_nan_and_coefficient_scale() -> None:
    """NaN cells stay blank and coefficients get the symmetric scale."""
    z = np.array([[-4.0, np.nan], [2.0, 1.0]])

    figure = create_heatmap(
        z=z,
        x=np.array([650.0, 651.0]),
        y=np.array([0.0, 0.5]),
        y_name="StepTime",
        z_name="coefficient",
        symmetric_range_quantile=1.0,
    )

    heatmap = figure.data[0]
    assert np.isnan(heatmap.z[0][1])
    assert list(heatmap.y) == [0.0, 0.5]
    assert figure.layout.yaxis.title.text == "StepTime"
    assert (figure.layout.coloraxis.cmin, figure.layout.coloraxis.cmax) == (-4.0, 4.0)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"z": np.zeros((1, 1)), "x": [1.0]},
        {"spectra": pl.DataFrame({"Time": [0.0]}), "z": np.zeros((1, 1))},
        {"z": np.zeros((2, 1)), "x": [1.0], "y": [0.0]},
    ],
)
def test_create_heatmap_rejects_ambiguous_or_misshaped_input(
    arguments: dict[str, object],
) -> None:
    """Exactly one of a frame or a matrix with matching axes is accepted."""
    with pytest.raises(ValueError):
        create_heatmap(**arguments)  # type: ignore[arg-type]
