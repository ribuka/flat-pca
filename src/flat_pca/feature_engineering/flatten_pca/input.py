"""Parquet input loading and validation for Flatten-PCA."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from flat_pca.spectral.schema import (
    METADATA_COLUMNS,
    parse_wavelength,
    select_wavelength_columns_in_range,
    wavelength_columns,
)
from flat_pca.utils import get_schema_from_polars

from ..preprocess.wavelength_filter import validate_wavelength_range

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


def validate_frame_schema(
    path: Path,
    frame: pl.DataFrame | pl.LazyFrame,
) -> frozenset[str]:
    """Validate one input frame's schema without reading any of its values.

    Checks the required-column, numeric-dtype, and wavelength-name-format
    rules that can be resolved from ``collect_schema()`` alone. Split out of
    ``validate_frame`` so the NumPy fast path can perform the same schema
    checks without also triggering ``validate_frame``'s separate value-scan
    ``collect()``; it instead validates values from the single read it
    already performs for preprocessing.

    Parameters
    ----------
    path : Path
        Source path used in validation errors.
    frame : pl.DataFrame | pl.LazyFrame
        Frame whose schema is validated. A ``LazyFrame`` is not materialized.

    Returns
    -------
    frozenset[str]
        Wavelength-column names.

    Raises
    ------
    ValueError
        If the frame violates the Flatten-PCA input schema rules.
    """
    schema = get_schema_from_polars(frame)
    columns = list(schema)
    missing_columns = [column for column in METADATA_COLUMNS if column not in schema]
    if missing_columns:
        raise ValueError(f"required columns missing in {path}: {missing_columns}")

    if any(not schema[column].is_numeric() for column in METADATA_COLUMNS):
        raise ValueError(f"metadata columns must be numeric in {path}")

    spectra = wavelength_columns(columns)
    if not spectra:
        raise ValueError(f"at least one wavelength column is required in {path}")
    for column in spectra:
        parse_wavelength(column)
    if any(not schema[column].is_numeric() for column in spectra):
        raise ValueError(f"spectral columns must be numeric in {path}")

    return frozenset(spectra)


def validate_frame(
    path: Path,
    frame: pl.LazyFrame,
    *,
    validate_metadata_uniqueness: bool = False,
    wavelength_range: tuple[float, float] | None = None,
) -> frozenset[str]:
    """Validate one input frame and return its cross-file comparison set.

    Parameters
    ----------
    path : Path
        Source path used in validation errors.
    frame : pl.LazyFrame
        Lazily scanned Parquet contents to validate.
    validate_metadata_uniqueness : bool, default False
        Whether to reject duplicate ``(Time, Step, Sequence)`` tuples.
    wavelength_range : tuple[float, float] | None, default None
        Inclusive ``(lower, upper)`` wavelength interval that a later
        ``apply_wavelength_range_filter`` call will keep, or ``None`` to keep
        every wavelength column. When given, the null/NaN/infinite-value
        check is limited to ``Time``, ``Step``, ``Sequence``, and wavelength
        columns within this range, so values in wavelength columns outside
        the requested range do not block processing. Schema checks (required
        columns, numeric dtypes, wavelength-name format, and the returned
        cross-file wavelength-set comparison) still cover every wavelength
        column regardless of ``wavelength_range``.

    Returns
    -------
    frozenset[str]
        Wavelength-column names.

    Raises
    ------
    ValueError
        If the frame violates the Flatten-PCA input schema or value rules.
    """
    spectra = validate_frame_schema(path, frame)
    columns = [*METADATA_COLUMNS, *spectra]

    if wavelength_range is None:
        value_check_columns = columns
    else:
        validated_range = validate_wavelength_range(wavelength_range)
        value_check_columns = list(METADATA_COLUMNS) + select_wavelength_columns_in_range(
            columns, validated_range
        )

    invalid_expressions = [
        pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
        for column in value_check_columns
    ]
    validation_expressions = [
        pl.any_horizontal(invalid_expressions).any().alias("has_invalid_value")
    ]
    if validate_metadata_uniqueness:
        validation_expressions.append(
            pl.struct(METADATA_COLUMNS)
            .is_duplicated()
            .any()
            .alias("has_duplicate_metadata")
        )
    validation = frame.select(validation_expressions).collect()
    has_invalid_value = validation["has_invalid_value"].item()
    if has_invalid_value:
        raise ValueError(f"input contains null, NaN, or infinite values: {path}")

    if validate_metadata_uniqueness:
        has_duplicate_metadata = validation["has_duplicate_metadata"].item()
        if has_duplicate_metadata:
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


def resolve_and_check_paths(
    paths: Sequence[str | Path],
    *,
    stem_uniqueness: StemUniquenessCheck = "skip",
) -> list[Path]:
    """Normalize input paths and apply the requested stem-uniqueness check.

    Extracted from ``load_and_validate_inputs`` so the NumPy fast path can
    reuse the same deterministic path resolution and stem-uniqueness rule
    without also running ``validate_frame``'s separate value-scan
    ``collect()`` for each path.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more input Parquet paths.
    stem_uniqueness : Literal["skip", "warn", "error"], default "skip"
        How to handle duplicate ``Path.stem`` values across inputs. See
        ``load_and_validate_inputs`` for details.

    Returns
    -------
    list[Path]
        Normalized absolute paths, sorted by path text.

    Raises
    ------
    ValueError
        If ``paths`` is empty, or ``stem_uniqueness="error"`` and stems
        repeat.
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

    return normalized_paths


def load_and_validate_inputs(
    paths: Sequence[str | Path],
    *,
    stem_uniqueness: StemUniquenessCheck = "skip",
    validate_metadata_uniqueness: bool = False,
    wavelength_range: tuple[float, float] | None = None,
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
    validate_metadata_uniqueness : bool, default False
        Whether to reject duplicate ``(Time, Step, Sequence)`` tuples in each
        input file.
    wavelength_range : tuple[float, float] | None, default None
        Inclusive ``(lower, upper)`` wavelength interval that a later
        ``apply_wavelength_range_filter`` call will keep, or ``None`` to keep
        every wavelength column. Forwarded to ``validate_frame`` to limit the
        null/NaN/infinite-value check to wavelength columns within this
        range; see ``validate_frame`` for details.

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
        input data violates the schema, value, requested uniqueness, or
        cross-file consistency requirements.
    """
    normalized_paths = resolve_and_check_paths(paths, stem_uniqueness=stem_uniqueness)

    loaded: list[tuple[Path, pl.LazyFrame]] = []
    expected_wavelengths: frozenset[str] | None = None
    for path in normalized_paths:
        frame = read_parquet(path)
        wavelengths = validate_frame(
            path,
            frame,
            validate_metadata_uniqueness=validate_metadata_uniqueness,
            wavelength_range=wavelength_range,
        )
        if expected_wavelengths is None:
            expected_wavelengths = wavelengths
        elif wavelengths != expected_wavelengths:
            raise ValueError("input wavelength sets must match")
        loaded.append((path, frame))

    return loaded
