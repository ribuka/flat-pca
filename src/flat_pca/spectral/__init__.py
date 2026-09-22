"""Shared spectral data conventions.

This package holds the column conventions of the spectral input format
itself, independent of any particular workflow. It is imported by both the
feature-engineering stages and the visualization helpers, and imports
nothing from them, so the two remain free of mutual dependencies.
"""

from .schema import (
    METADATA_COLUMNS,
    NON_SPECTRAL_COLUMNS,
    SOURCE_COLUMN,
    STEP_TIME_COLUMNS,
    flattened_feature_columns,
    parse_wavelength,
    select_wavelength_columns_in_range,
    wavelength_columns,
)

__all__ = [
    "METADATA_COLUMNS",
    "NON_SPECTRAL_COLUMNS",
    "SOURCE_COLUMN",
    "STEP_TIME_COLUMNS",
    "flattened_feature_columns",
    "parse_wavelength",
    "select_wavelength_columns_in_range",
    "wavelength_columns",
]
