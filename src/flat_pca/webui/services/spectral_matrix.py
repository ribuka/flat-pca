"""Spectral values of one ``(Step, Sequence)`` as a time × wavelength matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from flat_pca.spectral.schema import parse_wavelength, wavelength_columns


@dataclass(frozen=True)
class SpectralMatrix:
    """Values on a ``StepTime`` × wavelength grid.

    Attributes
    ----------
    values : np.ndarray
        ``float64`` matrix shaped ``(len(step_times), len(wavelengths))``.
        Missing cells are NaN.
    wavelengths : np.ndarray
        Ascending wavelengths of the columns.
    step_times : np.ndarray
        Ascending ``StepTime`` values of the rows.
    """

    values: np.ndarray
    wavelengths: np.ndarray
    step_times: np.ndarray


@dataclass(frozen=True)
class TrendLine:
    """Values along one axis of a ``SpectralMatrix``.

    Attributes
    ----------
    at : float
        Grid coordinate on the other axis that the line was cut at.
    x : np.ndarray
        Coordinates along the line.
    y : np.ndarray
        Values along the line.
    """

    at: float
    x: np.ndarray
    y: np.ndarray


def raw_segment_matrix(spectra: pl.DataFrame, step: int, sequence: int) -> SpectralMatrix:
    """Return the raw spectra of one ``(Step, Sequence)`` as a matrix.

    Parameters
    ----------
    spectra : pl.DataFrame
        One file's spectra with ``StepTime`` columns, as returned by
        ``display_cache.read_raw_spectra``.
    step, sequence : int
        Segment to extract.

    Returns
    -------
    SpectralMatrix
        Rows ordered by ``StepTime`` and wavelength columns in numeric order.

    Raises
    ------
    ValueError
        If the file has no row of the segment.
    """
    rows = spectra.filter(
        (pl.col("Step") == step) & (pl.col("Sequence") == sequence)
    ).sort("StepTime")
    if rows.is_empty():
        raise ValueError(f"no rows of (Step, Sequence) = ({step}, {sequence})")
    columns = sorted(wavelength_columns(rows.columns), key=parse_wavelength)
    return SpectralMatrix(
        values=rows.select(pl.col(columns).cast(pl.Float64)).to_numpy(),
        wavelengths=np.array([parse_wavelength(column) for column in columns]),
        step_times=rows["StepTime"].cast(pl.Float64).to_numpy(),
    )


def feature_segment_matrix(
    features: pl.DataFrame, values: np.ndarray, step: int, sequence: int
) -> SpectralMatrix:
    """Reshape one row of flattened feature values for one ``(Step, Sequence)``.

    Parameters
    ----------
    features : pl.DataFrame
        ``features.parquet`` of a run: ``wavelength``, ``Step``, ``Sequence``,
        and ``StepTime`` of each feature, in ``values`` order.
    values : np.ndarray
        One value per feature, such as an ``X.npy`` row or a component.
    step, sequence : int
        Segment to extract.

    Returns
    -------
    SpectralMatrix
        Grid of the segment's wavelengths and ``StepTime`` values. Grid
        points without a feature, and NaN values, are NaN.

    Raises
    ------
    ValueError
        If ``values`` does not have one value per feature, or no feature
        belongs to the segment.
    """
    if values.shape != (features.height,):
        raise ValueError(f"values are shaped {values.shape}, expected ({features.height},)")
    mask = (features["Step"] == step) & (features["Sequence"] == sequence)
    positions = np.flatnonzero(mask.to_numpy())
    if positions.size == 0:
        raise ValueError(f"no features of (Step, Sequence) = ({step}, {sequence})")
    wavelengths, columns = np.unique(
        features["wavelength"].to_numpy()[positions], return_inverse=True
    )
    step_times, rows = np.unique(
        features["StepTime"].to_numpy()[positions], return_inverse=True
    )
    matrix = np.full((step_times.size, wavelengths.size), np.nan)
    matrix[rows, columns] = np.asarray(values, dtype=np.float64)[positions]
    return SpectralMatrix(values=matrix, wavelengths=wavelengths, step_times=step_times)


def _nearest_index(axis: np.ndarray, value: float) -> int:
    """Return the index of the axis value closest to ``value``.

    Parameters
    ----------
    axis : np.ndarray
        Non-empty ascending axis.
    value : float
        Requested coordinate.

    Returns
    -------
    int
        Index of the nearest value; ties go to the smaller one.
    """
    return int(np.argmin(np.abs(axis - value)))


def trend_at_wavelength(matrix: SpectralMatrix, wavelength: float) -> TrendLine:
    """Cut the values over ``StepTime`` at the nearest wavelength.

    Parameters
    ----------
    matrix : SpectralMatrix
        Unbinned values.
    wavelength : float
        Requested wavelength.

    Returns
    -------
    TrendLine
        ``StepTime`` and the values of the nearest wavelength column.
    """
    column = _nearest_index(matrix.wavelengths, wavelength)
    return TrendLine(
        at=float(matrix.wavelengths[column]),
        x=matrix.step_times,
        y=matrix.values[:, column],
    )


def trend_at_step_time(matrix: SpectralMatrix, step_time: float) -> TrendLine:
    """Cut the values over wavelength at the nearest ``StepTime``.

    Parameters
    ----------
    matrix : SpectralMatrix
        Unbinned values.
    step_time : float
        Requested ``StepTime``.

    Returns
    -------
    TrendLine
        Wavelengths and the values of the nearest ``StepTime`` row.
    """
    row = _nearest_index(matrix.step_times, step_time)
    return TrendLine(
        at=float(matrix.step_times[row]), x=matrix.wavelengths, y=matrix.values[row]
    )
