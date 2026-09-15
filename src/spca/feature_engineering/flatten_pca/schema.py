"""Shared spectral schema definitions for the Flatten-PCA workflow."""

from math import isfinite

METADATA_COLUMNS = ("Time", "Step", "Sequence")


def parse_wavelength(column: str) -> float:
    """Parse and validate a canonical wavelength column name.

    Parameters
    ----------
    column : str
        Candidate wavelength column name.

    Returns
    -------
    float
        Finite wavelength encoded by the column name.

    Raises
    ------
    ValueError
        If the name is not the canonical ``f"{value:.1f}nm"`` representation.
    """
    if not column.endswith("nm"):
        raise ValueError(f"invalid wavelength column: {column!r}")
    try:
        wavelength = float(column[:-2])
    except ValueError as error:
        raise ValueError(f"invalid wavelength column: {column!r}") from error
    if not isfinite(wavelength) or f"{wavelength:.1f}nm" != column:
        raise ValueError(f"invalid wavelength column: {column!r}")
    return wavelength


def wavelength_columns(columns: list[str]) -> list[str]:
    """Return spectral columns from a complete Flatten-PCA column list.

    Parameters
    ----------
    columns : list[str]
        Complete input column names.

    Returns
    -------
    list[str]
        Column names not reserved for metadata.
    """
    return [column for column in columns if column not in METADATA_COLUMNS]
