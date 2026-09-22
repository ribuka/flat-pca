"""Spectral column conventions shared across the flat-pca workflows."""

from math import isfinite

SOURCE_COLUMN = "source"
METADATA_COLUMNS = ("Time", "Step", "Sequence")
STEP_TIME_COLUMNS = ("StepTime", "ReverseStepTime")
NON_SPECTRAL_COLUMNS = METADATA_COLUMNS + STEP_TIME_COLUMNS


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
        Column names not reserved for metadata or StepTime columns.
    """
    return [column for column in columns if column not in NON_SPECTRAL_COLUMNS]


def select_wavelength_columns_in_range(
    columns: list[str],
    wavelength_range: tuple[float, float],
) -> list[str]:
    """Return wavelength columns from a column list within an inclusive range.

    Parameters
    ----------
    columns : list[str]
        Complete input column names.
    wavelength_range : tuple[float, float]
        Already-validated, ordered, finite inclusive ``(lower, upper)``
        wavelength bounds.

    Returns
    -------
    list[str]
        Wavelength columns from ``columns`` whose parsed numeric value falls
        within ``wavelength_range``, in their existing relative order.
    """
    lower, upper = wavelength_range
    return [
        column
        for column in wavelength_columns(columns)
        if lower <= parse_wavelength(column) <= upper
    ]


def flattened_feature_columns(columns: list[str]) -> list[str]:
    """Return the feature columns of a flattened frame.

    Parameters
    ----------
    columns : list[str]
        Complete column names of a flattened frame, including
        ``SOURCE_COLUMN``.

    Returns
    -------
    list[str]
        Column names other than ``SOURCE_COLUMN``, in their existing order.
    """
    return [column for column in columns if column != SOURCE_COLUMN]
