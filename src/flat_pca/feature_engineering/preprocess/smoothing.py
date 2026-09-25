"""Time- and wavelength-direction smoothing for Flatten-PCA."""

import numpy as np
import polars as pl

from flat_pca.spectral.schema import parse_wavelength, wavelength_columns
from flat_pca.utils import get_columns_from_polars

from .ranges import validate_positive_finite


def apply_t_smoothing(
    frame: pl.DataFrame | pl.LazyFrame,
    t_smoothing_window: float | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Smooth spectral intensities within centered real-Time windows.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    t_smoothing_window : float | None
        Positive finite half-window width in the same units as ``Time``. If
        ``None``, smoothing is disabled.

    Returns
    -------
    pl.DataFrame
        Input rows and metadata with wavelength intensities replaced by the
        arithmetic mean in each closed Time window and metadata group.

    Raises
    ------
    ValueError
        If ``t_smoothing_window`` is not finite and greater than zero.
    """
    if t_smoothing_window is None:
        return frame
    window = validate_positive_finite(t_smoothing_window, "t_smoothing_window")

    if isinstance(frame, pl.LazyFrame):
        return frame.map_batches(
            lambda batch: _apply_t_smoothing_eager(batch, window),
            schema=frame.collect_schema(),
        )
    return _apply_t_smoothing_eager(frame, window)


def _apply_t_smoothing_eager(frame: pl.DataFrame, window: float) -> pl.DataFrame:
    """Apply centered Time smoothing to one materialized execution batch.

    Parameters
    ----------
    frame : pl.DataFrame
        One execution batch from a validated spectral frame.
    window : float
        Validated positive smoothing half-window.

    Returns
    -------
    pl.DataFrame
        Smoothed batch with its original schema and row order.
    """

    spectra = wavelength_columns(frame.columns)
    source_values = frame.select(spectra).to_numpy().astype(float, copy=False)
    smoothed_values = source_values.copy()
    times = frame["Time"].cast(pl.Float64).to_numpy()
    groups: dict[tuple[object, object], list[int]] = {}
    for row_index, group in enumerate(frame.select("Step", "Sequence").iter_rows()):
        groups.setdefault(group, []).append(row_index)

    for indices in groups.values():
        group_indices = np.asarray(indices)
        group_times = times[group_indices]
        group_values = source_values[group_indices]
        for local_index, row_index in enumerate(indices):
            in_window = np.abs(group_times - group_times[local_index]) <= window
            smoothed_values[row_index] = group_values[in_window].mean(axis=0)

    return frame.with_columns(
        pl.Series(column, smoothed_values[:, column_index])
        for column_index, column in enumerate(spectra)
    )


def _wavelength_window_indices(
    wavelengths: np.ndarray, window: float
) -> list[np.ndarray]:
    """Find, for each wavelength column, the columns within its closed window.

    Replaces an all-pairs scan (O(W^2) distance comparisons) with a sorted
    binary search (O(W log W)), which matters because building this LazyFrame
    expression runs once per input file. Wavelengths are one-dimensional real
    coordinates and the window predicate is a distance threshold, so the
    columns within any window always form one contiguous run once the
    wavelengths are sorted.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength (nm) parsed from each spectral column, in column order.
    window : float
        Positive finite half-window width in the same units as `wavelengths`.

    Returns
    -------
    list[np.ndarray]
        For each column index (matching `wavelengths` order), the indices of
        columns whose wavelength lies within the closed window, sorted back
        into original column order.
    """
    order = np.argsort(wavelengths, kind="stable")
    sorted_wavelengths = wavelengths[order]
    size = sorted_wavelengths.size

    lower = np.searchsorted(sorted_wavelengths, sorted_wavelengths - window, side="left")
    upper = np.searchsorted(sorted_wavelengths, sorted_wavelengths + window, side="right")

    # `searchsorted` compares against the rounded values `center - window` and
    # `center + window`, which can round differently than the direct
    # `abs(a - b) <= window` comparison the eager branch and callers rely on
    # (notably when `window` equals the sampling grid spacing). Correct each
    # boundary by at most one element against that direct comparison.
    result: list[np.ndarray] = [np.empty(0, dtype=int)] * size
    for position in range(size):
        center = sorted_wavelengths[position]
        low = int(lower[position])
        high = int(upper[position])
        if low > 0 and abs(sorted_wavelengths[low - 1] - center) <= window:
            low -= 1
        elif low < high and abs(sorted_wavelengths[low] - center) > window:
            low += 1
        if high < size and abs(sorted_wavelengths[high] - center) <= window:
            high += 1
        elif high > low and abs(sorted_wavelengths[high - 1] - center) > window:
            high -= 1
        result[order[position]] = np.sort(order[low:high])
    return result


def apply_w_smoothing(
    frame: pl.DataFrame | pl.LazyFrame,
    w_smoothing_window: float | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Smooth spectral intensities within centered real-wavelength windows.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    w_smoothing_window : float | None
        Positive finite half-window width in the same units as the wavelengths.
        If ``None``, smoothing is disabled.

    Returns
    -------
    pl.DataFrame
        Input rows and metadata with each wavelength intensity replaced by the
        arithmetic mean in its closed wavelength window.

    Raises
    ------
    ValueError
        If ``w_smoothing_window`` is not finite and greater than zero.
    """
    if w_smoothing_window is None:
        return frame
    window = validate_positive_finite(w_smoothing_window, "w_smoothing_window")

    columns = get_columns_from_polars(frame)
    spectra = wavelength_columns(columns)
    wavelengths = np.asarray([parse_wavelength(column) for column in spectra])
    if isinstance(frame, pl.LazyFrame):
        window_indices = _wavelength_window_indices(wavelengths, window)
        return frame.with_columns(
            pl.mean_horizontal(
                *(pl.col(spectra[index]) for index in indices)
            ).alias(target)
            for target, indices in zip(spectra, window_indices, strict=True)
        )
    source_values = frame.select(spectra).to_numpy().astype(float, copy=False)
    smoothed_values = source_values.copy()

    for column_index, wavelength in enumerate(wavelengths):
        in_window = np.abs(wavelengths - wavelength) <= window
        smoothed_values[:, column_index] = source_values[:, in_window].mean(axis=1)

    return frame.with_columns(
        pl.Series(column, smoothed_values[:, column_index])
        for column_index, column in enumerate(spectra)
    )
