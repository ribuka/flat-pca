"""Tests for reshaping spectra into matrices and cutting trends from them."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flat_pca.webui.services.spectral_matrix import (
    SpectralMatrix,
    feature_segment_matrix,
    raw_segment_matrix,
    trend_at_step_time,
    trend_at_wavelength,
)


def test_raw_segment_matrix_orders_rows_and_wavelengths() -> None:
    """Rows of one segment are ordered by StepTime and columns by wavelength."""
    spectra = pl.DataFrame(
        {
            "Time": [3.0, 1.0, 2.0, 0.0],
            "StepTime": [1.0, 0.0, 0.0, 0.0],
            "ReverseStepTime": [0.0, 1.0, 0.0, 0.0],
            "Step": [2, 2, 1, 1],
            "Sequence": [1, 1, 1, 0],
            "410.0nm": [4.0, 3.0, 2.0, 1.0],
            "400.0nm": [40.0, 30.0, 20.0, 10.0],
        }
    )

    matrix = raw_segment_matrix(spectra, 2, 1)

    assert matrix.wavelengths.tolist() == [400.0, 410.0]
    assert matrix.step_times.tolist() == [0.0, 1.0]
    assert matrix.values.tolist() == [[30.0, 3.0], [40.0, 4.0]]
    assert matrix.values.dtype == np.float64


def test_raw_segment_matrix_rejects_an_absent_segment() -> None:
    """A segment without rows is an error."""
    spectra = pl.DataFrame(
        {"Time": [0.0], "StepTime": [0.0], "Step": [1], "Sequence": [1], "400.0nm": [1.0]}
    )

    with pytest.raises(ValueError, match="no rows"):
        raw_segment_matrix(spectra, 1, 2)


@pytest.fixture
def features() -> pl.DataFrame:
    """Return features of two segments in a shuffled order.

    Segment ``(1, 1)`` lacks the feature at StepTime 0.5 and 410 nm, as if
    it had been dropped as a sparse column.
    """
    return pl.DataFrame(
        {
            "wavelength": [410.0, 400.0, 400.0, 400.0, 410.0],
            "Step": [1, 1, 1, 2, 1],
            "Sequence": [1, 1, 1, 1, 1],
            "StepTime": [0.0, 0.5, 0.0, 0.0, 1.0],
        }
    )


def test_feature_segment_matrix_reshapes_and_leaves_gaps(features: pl.DataFrame) -> None:
    """One row of feature values fills its segment's grid; gaps are NaN."""
    values = np.array([1.0, 2.0, np.nan, 9.0, 5.0], dtype=np.float32)

    matrix = feature_segment_matrix(features, values, 1, 1)

    assert matrix.wavelengths.tolist() == [400.0, 410.0]
    assert matrix.step_times.tolist() == [0.0, 0.5, 1.0]
    expected = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, 5.0]])
    np.testing.assert_array_equal(matrix.values, expected)
    assert matrix.values.dtype == np.float64


def test_feature_segment_matrix_validates_its_input(features: pl.DataFrame) -> None:
    """Values must match the features and the segment must exist."""
    with pytest.raises(ValueError, match="shaped"):
        feature_segment_matrix(features, np.zeros(4), 1, 1)
    with pytest.raises(ValueError, match="no features"):
        feature_segment_matrix(features, np.zeros(5), 3, 1)


def test_trends_cut_at_the_nearest_grid_point() -> None:
    """Trends use the nearest wavelength column and StepTime row."""
    matrix = SpectralMatrix(
        values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        wavelengths=np.array([400.0, 401.0, 405.0]),
        step_times=np.array([0.0, 2.0]),
    )

    by_time = trend_at_wavelength(matrix, 403.5)
    by_wavelength = trend_at_step_time(matrix, 1.2)

    assert by_time.at == 405.0
    assert by_time.x.tolist() == [0.0, 2.0]
    assert by_time.y.tolist() == [3.0, 6.0]
    assert by_wavelength.at == 2.0
    assert by_wavelength.x.tolist() == [400.0, 401.0, 405.0]
    assert by_wavelength.y.tolist() == [4.0, 5.0, 6.0]
