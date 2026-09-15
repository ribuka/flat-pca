"""Flatten-PCA input loading and validation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import polars as pl

_METADATA_COLUMNS = ("Time", "Step", "Sequence")


def _parse_wavelength(column: str) -> float:
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


def _read_parquet(path: Path) -> pl.DataFrame:
    """Read one validated filesystem path as Parquet.

    Parameters
    ----------
    path : Path
        Normalized absolute input path.

    Returns
    -------
    pl.DataFrame
        Eagerly loaded Parquet contents.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the path is not a file or cannot be read as Parquet.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"failed to read Parquet input: {path}") from error


def _validate_frame(
    path: Path,
    frame: pl.DataFrame,
) -> tuple[frozenset[str], frozenset[tuple[object, ...]]]:
    """Validate one input frame and return its cross-file comparison sets.

    Parameters
    ----------
    path : Path
        Source path used in validation errors.
    frame : pl.DataFrame
        Parquet contents to validate.

    Returns
    -------
    tuple[frozenset[str], frozenset[tuple[object, ...]]]
        Wavelength-column names and metadata-key tuples.

    Raises
    ------
    ValueError
        If the frame violates the Flatten-PCA input schema or value rules.
    """
    missing_columns = [
        column for column in _METADATA_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"required columns missing in {path}: {missing_columns}")

    schema = frame.schema
    if any(not schema[column].is_numeric() for column in _METADATA_COLUMNS):
        raise ValueError(f"metadata columns must be numeric in {path}")

    wavelength_columns = [
        column for column in frame.columns if column not in _METADATA_COLUMNS
    ]
    if not wavelength_columns:
        raise ValueError(f"at least one wavelength column is required in {path}")
    for column in wavelength_columns:
        _parse_wavelength(column)
    if any(not schema[column].is_numeric() for column in wavelength_columns):
        raise ValueError(f"spectral columns must be numeric in {path}")

    invalid_expressions = [
        pl.col(column).is_null()
        | ~pl.col(column).cast(pl.Float64).is_finite()
        for column in frame.columns
    ]
    has_invalid_value = frame.select(
        pl.any_horizontal(invalid_expressions).any()
    ).item()
    if has_invalid_value:
        raise ValueError(f"input contains null, NaN, or infinite values: {path}")

    metadata = frame.select(_METADATA_COLUMNS)
    if metadata.is_duplicated().any():
        raise ValueError(f"input contains duplicate metadata keys: {path}")

    return frozenset(wavelength_columns), frozenset(metadata.iter_rows())


def _load_and_validate_inputs(
    paths: Sequence[str | Path],
) -> list[tuple[Path, pl.DataFrame]]:
    """Load and validate Flatten-PCA Parquet inputs deterministically.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more input Parquet paths.

    Returns
    -------
    list[tuple[Path, pl.DataFrame]]
        Normalized absolute paths and their frames, sorted by path text.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If paths are empty, stems repeat, or input data violates the schema,
        value, uniqueness, or cross-file consistency requirements.
    """
    if isinstance(paths, (str, Path)) or len(paths) == 0:
        raise ValueError("paths must contain at least one input path")

    normalized_paths = sorted(
        (Path(path).resolve() for path in paths),
        key=str,
    )
    stems = [path.stem for path in normalized_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("input path stems must be unique")

    loaded: list[tuple[Path, pl.DataFrame]] = []
    expected_wavelengths: frozenset[str] | None = None
    expected_metadata_keys: frozenset[tuple[object, ...]] | None = None
    for path in normalized_paths:
        frame = _read_parquet(path)
        wavelengths, metadata_keys = _validate_frame(path, frame)
        if expected_wavelengths is None:
            expected_wavelengths = wavelengths
            expected_metadata_keys = metadata_keys
        else:
            if wavelengths != expected_wavelengths:
                raise ValueError("input wavelength sets must match")
            if metadata_keys != expected_metadata_keys:
                raise ValueError("input metadata-key sets must match")
        loaded.append((path, frame))

    return loaded
