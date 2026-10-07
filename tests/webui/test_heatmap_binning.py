"""Tests for binning heatmap rows along StepTime."""

from __future__ import annotations

import numpy as np

from flat_pca.webui.services.heatmap_binning import bin_step_times
from flat_pca.webui.services.spectral_matrix import SpectralMatrix


def _matrix(values: np.ndarray, step_times: list[float]) -> SpectralMatrix:
    """Build a matrix with one wavelength per column."""
    return SpectralMatrix(
        values=values,
        wavelengths=np.arange(values.shape[1], dtype=np.float64) + 400.0,
        step_times=np.array(step_times),
    )


def test_matrix_within_the_limit_is_not_binned() -> None:
    """A matrix with at most max_cells cells is returned as is."""
    matrix = _matrix(np.ones((3, 2)), [0.0, 1.0, 2.0])

    binned = bin_step_times(matrix, max_cells=6)

    assert binned.binned is False
    assert binned.matrix is matrix
    assert binned.source_rows == 3


def test_rows_are_averaged_over_equal_width_bins() -> None:
    """Rows are averaged per StepTime bin, ignoring NaN; wavelengths are kept."""
    values = np.array(
        [[1.0, 10.0], [3.0, np.nan], [5.0, 50.0], [7.0, 70.0], [9.0, np.nan]]
    )
    # Range 0..4 in two bins: [0, 2) holds 0, 1; [2, 4] holds 2, 3, 4.
    matrix = _matrix(values, [0.0, 1.0, 2.0, 3.0, 4.0])

    binned = bin_step_times(matrix, max_cells=5)

    assert binned.binned is True
    assert binned.source_rows == 5
    np.testing.assert_array_equal(binned.matrix.wavelengths, matrix.wavelengths)
    np.testing.assert_allclose(binned.matrix.step_times, [0.5, 3.0])
    np.testing.assert_allclose(binned.matrix.values, [[2.0, 10.0], [7.0, 60.0]])


def test_empty_bins_are_dropped_and_all_nan_cells_stay_nan() -> None:
    """Bins without rows disappear; a bin column without values is NaN."""
    values = np.array([[1.0], [np.nan], [3.0]])
    # Range 0..10 in two bins: 0 and 1 fall in the first, 10 in the second.
    matrix = _matrix(values, [0.0, 1.0, 10.0])

    binned = bin_step_times(matrix, max_cells=2)

    np.testing.assert_allclose(binned.matrix.step_times, [0.5, 10.0])
    np.testing.assert_allclose(binned.matrix.values, [[1.0], [3.0]])

    gaps = bin_step_times(_matrix(np.array([[np.nan], [np.nan], [1.0]]), [0.0, 1.0, 2.0]), 2)
    assert np.isnan(gaps.matrix.values[0, 0])
