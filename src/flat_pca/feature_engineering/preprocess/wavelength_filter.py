"""Wavelength-range column filtering for the Flatten-PCA workflow."""

from __future__ import annotations

from math import isfinite

import polars as pl

from ..flatten_pca.schema import select_wavelength_columns_in_range, wavelength_columns


def validate_wavelength_range(value: tuple[float, float]) -> tuple[float, float]:
    """Validate and coerce an inclusive ``wavelength_range`` interval.

    Parameters
    ----------
    value : tuple[float, float]
        Candidate lower and upper bounds.

    Returns
    -------
    tuple[float, float]
        Finite, ordered floating-point bounds.

    Raises
    ------
    ValueError
        If the candidate is malformed, nonfinite, or reversed.
    """
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(bound, bool) for bound in value)
    ):
        raise ValueError("wavelength_range must contain two finite bounds")
    try:
        lower, upper = (float(bound) for bound in value)
    except (TypeError, ValueError) as error:
        raise ValueError("wavelength_range must contain two finite bounds") from error
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError("wavelength_range must contain ordered finite bounds")
    return lower, upper


def apply_wavelength_range_filter(
    frame: pl.DataFrame | pl.LazyFrame,
    wavelength_range: tuple[float, float] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Keep only wavelength columns within an inclusive wavelength interval.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Validated Flatten-PCA input frame.
    wavelength_range : tuple[float, float] | None
        Inclusive ``(lower, upper)`` wavelength interval, or ``None`` to keep
        every wavelength column.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Frame containing all non-spectral columns and only the wavelength
        columns whose numeric value falls within ``wavelength_range``, or the
        input frame unchanged if ``wavelength_range`` is ``None``.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite, or if no
        wavelength column falls within it.
    """
    if wavelength_range is None:
        return frame
    validated_range = validate_wavelength_range(wavelength_range)

    columns = _column_names(frame)
    spectra = wavelength_columns(columns)
    non_spectral_columns = [column for column in columns if column not in spectra]
    selected_columns = select_wavelength_columns_in_range(columns, validated_range)
    if not selected_columns:
        raise ValueError("wavelength_range matches no wavelength columns")
    return frame.select(*non_spectral_columns, *selected_columns)


def _column_names(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return column names without materializing a LazyFrame.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Spectral frame whose schema supplies the column names.

    Returns
    -------
    list[str]
        Frame column names in their existing order.
    """
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns
