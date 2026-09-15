"""Time- and wavelength-direction smoothing for Flatten-PCA."""

from math import isfinite

import numpy as np
import polars as pl

from .schema import parse_wavelength, wavelength_columns


def apply_t_smoothing(
    frame: pl.DataFrame,
    t_smoothing_window: float | None,
) -> pl.DataFrame:
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
    try:
        window = float(t_smoothing_window)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "t_smoothing_window must be finite and greater than 0"
        ) from error
    if isinstance(t_smoothing_window, bool) or not isfinite(window) or window <= 0:
        raise ValueError("t_smoothing_window must be finite and greater than 0")

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


def apply_w_smoothing(
    frame: pl.DataFrame,
    w_smoothing_window: float | None,
) -> pl.DataFrame:
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
    try:
        window = float(w_smoothing_window)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "w_smoothing_window must be finite and greater than 0"
        ) from error
    if isinstance(w_smoothing_window, bool) or not isfinite(window) or window <= 0:
        raise ValueError("w_smoothing_window must be finite and greater than 0")

    spectra = wavelength_columns(frame.columns)
    wavelengths = np.asarray([parse_wavelength(column) for column in spectra])
    source_values = frame.select(spectra).to_numpy().astype(float, copy=False)
    smoothed_values = source_values.copy()

    for column_index, wavelength in enumerate(wavelengths):
        in_window = np.abs(wavelengths - wavelength) <= window
        smoothed_values[:, column_index] = source_values[:, in_window].mean(axis=1)

    return frame.with_columns(
        pl.Series(column, smoothed_values[:, column_index])
        for column_index, column in enumerate(spectra)
    )
