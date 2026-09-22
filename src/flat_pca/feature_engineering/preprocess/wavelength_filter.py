"""Wavelength-range column filtering for the Flatten-PCA workflow."""

from __future__ import annotations

import polars as pl

from flat_pca.spectral.schema import (
    select_wavelength_columns_in_range,
    wavelength_columns,
)
from flat_pca.utils import get_columns_from_polars

from .ranges import validate_ordered_range


def validate_wavelength_range(value: tuple[float, float]) -> tuple[float, float]:
    """Validate and coerce an inclusive ``wavelength_range`` interval.

    Thin wrapper over ``validate_ordered_range`` that fixes the public
    argument name used in validation messages.

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
    return validate_ordered_range(value, "wavelength_range")


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

    columns = get_columns_from_polars(frame)
    spectra = wavelength_columns(columns)
    non_spectral_columns = [column for column in columns if column not in spectra]
    selected_columns = select_wavelength_columns_in_range(columns, validated_range)
    if not selected_columns:
        raise ValueError("wavelength_range matches no wavelength columns")
    return frame.select(*non_spectral_columns, *selected_columns)
