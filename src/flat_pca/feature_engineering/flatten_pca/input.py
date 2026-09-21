"""Parquet input loading and validation for Flatten-PCA."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from .schema import METADATA_COLUMNS, parse_wavelength, wavelength_columns

StemUniquenessCheck = Literal["skip", "warn", "error"]


def read_parquet(path: Path) -> pl.LazyFrame:
    """Read one validated filesystem path as Parquet.

    Parameters
    ----------
    path : Path
        Normalized absolute input path.

    Returns
    -------
    pl.LazyFrame
        Lazily scanned Parquet contents.

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
        return pl.scan_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"failed to read Parquet input: {path}") from error


def validate_frame(
    path: Path,
    frame: pl.DataFrame,
) -> frozenset[str]:
    """Validate one input frame and return its cross-file comparison set.

    Parameters
    ----------
    path : Path
        Source path used in validation errors.
    frame : pl.DataFrame
        Parquet contents to validate.

    Returns
    -------
    frozenset[str]
        Wavelength-column names.

    Raises
    ------
    ValueError
        If the frame violates the Flatten-PCA input schema or value rules.
    """
    missing_columns = [
        column for column in METADATA_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"required columns missing in {path}: {missing_columns}")

    schema = frame.schema
    if any(not schema[column].is_numeric() for column in METADATA_COLUMNS):
        raise ValueError(f"metadata columns must be numeric in {path}")

    spectra = wavelength_columns(frame.columns)
    if not spectra:
        raise ValueError(f"at least one wavelength column is required in {path}")
    for column in spectra:
        parse_wavelength(column)
    if any(not schema[column].is_numeric() for column in spectra):
        raise ValueError(f"spectral columns must be numeric in {path}")

    invalid_expressions = [
        pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
        for column in frame.columns
    ]
    has_invalid_value = frame.select(
        pl.any_horizontal(invalid_expressions).any()
    ).item()
    if has_invalid_value:
        raise ValueError(f"input contains null, NaN, or infinite values: {path}")

    metadata = frame.select(METADATA_COLUMNS)
    if metadata.is_duplicated().any():
        raise ValueError(f"input contains duplicate metadata keys: {path}")

    return frozenset(spectra)


def validate_metadata_alignment(
    inputs: Sequence[tuple[Path, pl.DataFrame | pl.LazyFrame]],
) -> None:
    """Validate that Time, Step, and Sequence combinations match across inputs.

    This check is independent of ``load_and_validate_inputs`` so it can be
    run after row-filtering stages (for example ``filter_target_steps``),
    comparing only the combinations that survive filtering.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame | pl.LazyFrame]]
        Paths paired with frames to compare.

    Raises
    ------
    ValueError
        If the frames do not share an identical set of ``(Time, Step,
        Sequence)`` tuples.
    """
    expected_metadata_keys: frozenset[tuple[object, ...]] | None = None
    for _, frame in inputs:
        collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        metadata_keys = frozenset(collected.select(METADATA_COLUMNS).iter_rows())
        if expected_metadata_keys is None:
            expected_metadata_keys = metadata_keys
        elif metadata_keys != expected_metadata_keys:
            raise ValueError("input metadata-key sets must match")


def load_and_validate_inputs(
    paths: Sequence[str | Path],
    *,
    stem_uniqueness: StemUniquenessCheck = "skip",
) -> list[tuple[Path, pl.LazyFrame]]:
    """Load and validate Flatten-PCA Parquet inputs deterministically.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more input Parquet paths.
    stem_uniqueness : Literal["skip", "warn", "error"], default "skip"
        How to handle duplicate ``Path.stem`` values across inputs.
        ``"skip"`` performs no check, ``"warn"`` emits a ``UserWarning`` and
        continues, and ``"error"`` raises ``ValueError``. The ``source``
        column produced downstream is derived from the full normalized path,
        not the stem, so stem uniqueness is not required for correctness.

    Returns
    -------
    list[tuple[Path, pl.LazyFrame]]
        Normalized absolute paths and lazily scanned frames, sorted by path
        text.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If paths are empty, ``stem_uniqueness="error"`` and stems repeat, or
        input data violates the schema, value, uniqueness, or cross-file
        consistency requirements.
    """
    if isinstance(paths, (str, Path)) or len(paths) == 0:
        raise ValueError("paths must contain at least one input path")

    normalized_paths = sorted((Path(path).resolve() for path in paths), key=str)
    if stem_uniqueness != "skip":
        stems = [path.stem for path in normalized_paths]
        if len(stems) != len(set(stems)):
            if stem_uniqueness == "error":
                raise ValueError("input path stems must be unique")
            warnings.warn("input path stems are not unique", stacklevel=2)

    loaded: list[tuple[Path, pl.LazyFrame]] = []
    expected_wavelengths: frozenset[str] | None = None
    for path in normalized_paths:
        frame = read_parquet(path)
        # Validation requires concrete values, but processing remains lazy.
        wavelengths = validate_frame(path, frame.collect())
        if expected_wavelengths is None:
            expected_wavelengths = wavelengths
        elif wavelengths != expected_wavelengths:
            raise ValueError("input wavelength sets must match")
        loaded.append((path, frame))

    return loaded
