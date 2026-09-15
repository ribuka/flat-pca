"""Flatten-PCA input loading and validation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl

_METADATA_COLUMNS = ("Time", "Step", "Sequence")


def _flatten_inputs(
    inputs: Sequence[tuple[Path, pl.DataFrame]],
) -> pl.DataFrame:
    """Flatten each validated spectral frame into one deterministic row.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Normalized input paths paired with validated spectral frames.

    Returns
    -------
    pl.DataFrame
        One row per input file, ordered by normalized path, with ``filename``
        followed by spectral features ordered by numeric wavelength, Step,
        Sequence, and Time.

    Raises
    ------
    ValueError
        If no inputs are provided, formatted feature names collide, or frames
        do not produce the same ordered feature names.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must contain at least one validated frame")

    rows: list[dict[str, object]] = []
    expected_feature_names: list[str] | None = None
    for path, frame in sorted(inputs, key=lambda item: str(item[0].resolve())):
        wavelength_columns = sorted(
            (
                (_parse_wavelength(column), column)
                for column in frame.columns
                if column not in _METADATA_COLUMNS
            ),
            key=lambda item: item[0],
        )
        sorted_frame = frame.sort("Step", "Sequence", "Time")
        feature_names: list[str] = []
        feature_values: list[object] = []
        for _, wavelength_column in wavelength_columns:
            for step, sequence, time, intensity in sorted_frame.select(
                "Step",
                "Sequence",
                "Time",
                wavelength_column,
            ).iter_rows():
                feature_names.append(
                    f"{wavelength_column}*{int(step)}*{int(sequence)}_"
                    f"{float(time):.2f}nm"
                )
                feature_values.append(intensity)

        if len(feature_names) != len(set(feature_names)):
            raise ValueError("duplicate flattened feature names")
        if expected_feature_names is None:
            expected_feature_names = feature_names
        elif feature_names != expected_feature_names:
            raise ValueError("input frames must produce matching flattened features")

        row: dict[str, object] = {"filename": path.stem}
        row.update(zip(feature_names, feature_values, strict=True))
        rows.append(row)

    return pl.DataFrame(rows).select("filename", *expected_feature_names)


def _apply_w_normalization(
    frame: pl.DataFrame,
    w_normalization_range: tuple[float, float] | None,
) -> pl.DataFrame:
    """Normalize spectra by a mean from an inclusive wavelength interval.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    w_normalization_range : tuple[float, float] | None
        Inclusive reference interval in the same units as the wavelengths. If
        ``None``, normalization is disabled.

    Returns
    -------
    pl.DataFrame
        Input rows and metadata with each row's intensities divided by its
        reference-wavelength mean.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; the interval has no
        wavelengths; or a row's reference mean is zero or nonfinite.
    """
    if w_normalization_range is None:
        return frame
    if (
        not isinstance(w_normalization_range, tuple)
        or len(w_normalization_range) != 2
        or any(isinstance(bound, bool) for bound in w_normalization_range)
    ):
        raise ValueError("w_normalization_range must contain two finite bounds")
    try:
        lower, upper = (float(bound) for bound in w_normalization_range)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "w_normalization_range must contain two finite bounds"
        ) from error
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(
            "w_normalization_range must contain ordered finite bounds"
        )

    wavelength_columns = [
        column for column in frame.columns if column not in _METADATA_COLUMNS
    ]
    wavelengths = np.asarray([_parse_wavelength(column) for column in wavelength_columns])
    in_reference = (wavelengths >= lower) & (wavelengths <= upper)
    if not in_reference.any():
        raise ValueError("w_normalization_range reference interval is empty")

    source_values = frame.select(wavelength_columns).to_numpy().astype(
        float,
        copy=False,
    )
    reference_mean = source_values[:, in_reference].mean(axis=1)
    if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
        raise ValueError(
            "w_normalization_range reference mean must be finite and nonzero"
        )
    normalized_values = source_values / reference_mean[:, np.newaxis]

    return frame.with_columns(
        pl.Series(column, normalized_values[:, column_index])
        for column_index, column in enumerate(wavelength_columns)
    )


def _apply_t_normalization(
    frame: pl.DataFrame,
    t_normalization_range: tuple[float, float] | None,
) -> pl.DataFrame:
    """Normalize spectra by a mean from an inclusive real-Time interval.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    t_normalization_range : tuple[float, float] | None
        Inclusive reference interval in the same units as ``Time``. If
        ``None``, normalization is disabled.

    Returns
    -------
    pl.DataFrame
        Input rows and metadata with intensities divided by the reference mean
        for each Step, Sequence, and wavelength.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; a metadata group has
        no reference observations; or a reference mean is zero or nonfinite.
    """
    if t_normalization_range is None:
        return frame
    if (
        not isinstance(t_normalization_range, tuple)
        or len(t_normalization_range) != 2
        or any(isinstance(bound, bool) for bound in t_normalization_range)
    ):
        raise ValueError("t_normalization_range must contain two finite bounds")
    try:
        lower, upper = (float(bound) for bound in t_normalization_range)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "t_normalization_range must contain two finite bounds"
        ) from error
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(
            "t_normalization_range must contain ordered finite bounds"
        )

    wavelength_columns = [
        column for column in frame.columns if column not in _METADATA_COLUMNS
    ]
    source_values = frame.select(wavelength_columns).to_numpy().astype(
        float,
        copy=False,
    )
    normalized_values = source_values.copy()
    times = frame["Time"].cast(pl.Float64).to_numpy()
    groups: dict[tuple[object, object], list[int]] = {}
    for row_index, group in enumerate(frame.select("Step", "Sequence").iter_rows()):
        groups.setdefault(group, []).append(row_index)
    if not groups:
        raise ValueError(
            "t_normalization_range reference interval is empty for a group"
        )

    for indices in groups.values():
        group_indices = np.asarray(indices)
        in_reference = (times[group_indices] >= lower) & (
            times[group_indices] <= upper
        )
        if not in_reference.any():
            raise ValueError(
                "t_normalization_range reference interval is empty for a group"
            )
        reference_mean = source_values[group_indices[in_reference]].mean(axis=0)
        if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
            raise ValueError(
                "t_normalization_range reference mean must be finite and nonzero"
            )
        normalized_values[group_indices] = (
            source_values[group_indices] / reference_mean
        )

    return frame.with_columns(
        pl.Series(column, normalized_values[:, column_index])
        for column_index, column in enumerate(wavelength_columns)
    )


def _apply_t_smoothing(
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

    wavelength_columns = [
        column for column in frame.columns if column not in _METADATA_COLUMNS
    ]
    source_values = frame.select(wavelength_columns).to_numpy().astype(
        float,
        copy=False,
    )
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
        for column_index, column in enumerate(wavelength_columns)
    )


def _apply_w_smoothing(
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

    wavelength_columns = [
        column for column in frame.columns if column not in _METADATA_COLUMNS
    ]
    wavelengths = np.asarray([_parse_wavelength(column) for column in wavelength_columns])
    source_values = frame.select(wavelength_columns).to_numpy().astype(
        float,
        copy=False,
    )
    smoothed_values = source_values.copy()

    for column_index, wavelength in enumerate(wavelengths):
        in_window = np.abs(wavelengths - wavelength) <= window
        smoothed_values[:, column_index] = source_values[:, in_window].mean(axis=1)

    return frame.with_columns(
        pl.Series(column, smoothed_values[:, column_index])
        for column_index, column in enumerate(wavelength_columns)
    )


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
