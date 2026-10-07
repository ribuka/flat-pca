"""Time-axis binning that keeps heatmaps within the cell limit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_matrix import SpectralMatrix


@dataclass(frozen=True)
class BinnedMatrix:
    """A matrix prepared for the heatmap, possibly averaged over time bins.

    Attributes
    ----------
    matrix : SpectralMatrix
        Values to draw. When binned, each row is the mean of one time bin and
        its ``StepTime`` is the mean ``StepTime`` of the rows in the bin.
    binned : bool
        Whether rows were averaged.
    source_rows : int
        Number of rows before binning.
    """

    matrix: SpectralMatrix
    binned: bool
    source_rows: int


def bin_step_times(matrix: SpectralMatrix, max_cells: int) -> BinnedMatrix:
    """Average rows over equal-width ``StepTime`` bins above a cell limit.

    The wavelength axis is never reduced. When the matrix has more than
    ``max_cells`` cells, the ``StepTime`` range is split into
    ``max(1, max_cells // n_wavelengths)`` equal-width bins and each bin's
    rows are averaged, ignoring NaN. Bins without rows are dropped.

    Parameters
    ----------
    matrix : SpectralMatrix
        Unbinned values.
    max_cells : int
        Maximum number of cells sent to the browser.

    Returns
    -------
    BinnedMatrix
        ``matrix`` itself when it fits, otherwise the binned matrix.
    """
    n_rows, n_columns = matrix.values.shape
    if n_rows * n_columns <= max_cells:
        return BinnedMatrix(matrix=matrix, binned=False, source_rows=n_rows)
    n_bins = max(1, max_cells // max(n_columns, 1))
    edges = np.linspace(matrix.step_times[0], matrix.step_times[-1], n_bins + 1)
    bins = np.clip(np.searchsorted(edges, matrix.step_times, side="right") - 1, 0, n_bins - 1)
    present = ~np.isnan(matrix.values)
    sums = np.zeros((n_bins, n_columns))
    counts = np.zeros((n_bins, n_columns))
    np.add.at(sums, bins, np.where(present, matrix.values, 0.0))
    np.add.at(counts, bins, present)
    rows_per_bin = np.bincount(bins, minlength=n_bins)
    times = np.bincount(bins, weights=matrix.step_times, minlength=n_bins)
    kept = rows_per_bin > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    binned = SpectralMatrix(
        values=means[kept],
        wavelengths=matrix.wavelengths,
        step_times=times[kept] / rows_per_bin[kept],
    )
    return BinnedMatrix(matrix=binned, binned=True, source_rows=n_rows)
