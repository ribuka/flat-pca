"""NumPy fast path for the ``materialize_once=True`` preprocessing default.

Replaces the double Parquet read (once in ``validate_frame`` for value
validation, once more for preprocessing) and the practice of smoothing every
row and wavelength column before most of them are discarded by
downsampling. Instead, each file is read once and its per-row and
per-wavelength-column smoothing output is computed only where later stages
still need it: the rows and columns that survive downsampling, plus any
extra rows or columns required only to compute a normalization reference
mean. Every stage below reuses the exact boolean-window and grouping
arithmetic of the corresponding ``preprocess`` module (``apply_t_smoothing``,
``apply_w_smoothing``, ``apply_t_normalization``, ``apply_w_normalization``),
just restricted to the positions that matter, so results are bit-identical
to the ``materialize_once=False`` deferred path, not merely close to it.

``materialize_once=False`` keeps using the original polars implementation in
``preprocess`` unchanged; this module is only used by
``preprocess_and_flatten``'s ``materialize_once=True`` branch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from flat_pca.spectral.schema import (
    METADATA_COLUMNS,
    parse_wavelength,
    select_wavelength_columns_in_range,
    wavelength_columns,
)

from ..preprocess.downsampling import collect_unique_times, resolve_downsampled_values
from ..preprocess.filter import filter_target_steps
from ..preprocess.ranges import validate_ordered_range, validate_positive_finite
from ..preprocess.step_time import add_step_time_columns
from ..preprocess.trim import apply_edge_trim
from ..preprocess.wavelength_filter import validate_wavelength_range
from .input import read_parquet, validate_frame_schema
from .input import validate_metadata_alignment as _validate_metadata_alignment

MetadataGroups = dict[tuple[object, object], list[int]]


def all_wavelength_columns_are_float(
    paths: Sequence[Path], wavelength_range: tuple[float, float] | None
) -> bool:
    """Report whether every file's wavelength_range-restricted columns are float.

    The fast path converts spectral values to ``float64`` throughout, which
    reproduces ``Float32``/``Float64`` input exactly but would silently
    round an integer beyond 53 bits or widen a decimal dtype -- exactly the
    dtype rule ``flatten.py``'s ``_has_float_spectral_columns`` already
    enforces for the flatten stage itself. Checked from schema alone (no
    data read), so this is cheap to call before deciding which
    ``preprocess_and_flatten`` ``materialize_once=True`` implementation to
    use.

    Parameters
    ----------
    paths : Sequence[Path]
        Resolved input paths.
    wavelength_range : tuple[float, float] | None
        Inclusive ``(lower, upper)`` wavelength interval that will restrict
        which columns are read, or ``None`` to check every wavelength
        column.

    Returns
    -------
    bool
        ``True`` if every checked wavelength column, in every file, has a
        floating-point dtype.
    """
    validated_range = (
        validate_wavelength_range(wavelength_range) if wavelength_range is not None else None
    )
    for path in paths:
        schema = read_parquet(path).collect_schema()
        candidate_columns = wavelength_columns(schema.names())
        if validated_range is not None:
            candidate_columns = select_wavelength_columns_in_range(
                candidate_columns, validated_range
            )
        if any(not schema[column].is_float() for column in candidate_columns):
            return False
    return True


def _read_file_metadata(path: Path, target_steps: list[int] | None) -> pl.DataFrame:
    """Read one file's ``(Time, Step, Sequence)`` metadata, applying ``target_steps``.

    Parameters
    ----------
    path : Path
        Normalized absolute input path.
    target_steps : list[int] | None
        ``Step`` values to keep, or ``None`` to keep every row.

    Returns
    -------
    pl.DataFrame
        Cheap metadata-only frame, projected to three columns so the shared
        Time grid can be resolved before any spectral column is read.
    """
    frame = read_parquet(path).select("Time", "Step", "Sequence")
    filtered = filter_target_steps(frame, target_steps)
    assert isinstance(filtered, pl.LazyFrame)
    return filtered.collect()


def _prepare_file_metadata(
    metadata: pl.DataFrame, edge_trim: Sequence[float] | None
) -> pl.DataFrame:
    """Add StepTime/ReverseStepTime and apply edge trimming to file metadata.

    Parameters
    ----------
    metadata : pl.DataFrame
        Target-step-filtered ``(Time, Step, Sequence)`` metadata.
    edge_trim : Sequence[float] | None
        Edge-trim thresholds forwarded to ``apply_edge_trim``.

    Returns
    -------
    pl.DataFrame
        Metadata with StepTime/ReverseStepTime added and edge-trimmed rows
        dropped, sorted by Time ascending.
    """
    with_step_time = add_step_time_columns(metadata)
    assert isinstance(with_step_time, pl.DataFrame)
    trimmed = apply_edge_trim(with_step_time, edge_trim)
    assert isinstance(trimmed, pl.DataFrame)
    return trimmed


def _validate_values(
    path: Path, frame: pl.DataFrame, value_check_columns: list[str]
) -> None:
    """Reject null, NaN, or infinite values in the given columns.

    Mirrors ``validate_frame``'s value check exactly, run once on the same
    read this module already performs for preprocessing instead of on a
    separate ``collect()``.

    Parameters
    ----------
    path : Path
        Source path used in the error message.
    frame : pl.DataFrame
        Freshly read, not-yet-filtered file contents.
    value_check_columns : list[str]
        Column names to check.

    Raises
    ------
    ValueError
        If any checked column contains a null, NaN, or infinite value.
    """
    invalid_expressions = [
        pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
        for column in value_check_columns
    ]
    has_invalid_value = frame.select(
        pl.any_horizontal(invalid_expressions).any().alias("has_invalid_value")
    )["has_invalid_value"].item()
    if has_invalid_value:
        raise ValueError(f"input contains null, NaN, or infinite values: {path}")


def _validate_no_duplicate_metadata(path: Path, frame: pl.DataFrame) -> None:
    """Reject duplicate ``(Time, Step, Sequence)`` tuples.

    Mirrors ``validate_frame``'s optional duplicate-key check.

    Parameters
    ----------
    path : Path
        Source path used in the error message.
    frame : pl.DataFrame
        Freshly read, not-yet-filtered file contents.

    Raises
    ------
    ValueError
        If any ``(Time, Step, Sequence)`` tuple repeats.
    """
    has_duplicate_metadata = frame.select(
        pl.struct(METADATA_COLUMNS).is_duplicated().any().alias("has_duplicate_metadata")
    )["has_duplicate_metadata"].item()
    if has_duplicate_metadata:
        raise ValueError(f"input contains duplicate metadata keys: {path}")


def _group_row_indices(
    steps: Sequence[object], sequences: Sequence[object]
) -> MetadataGroups:
    """Group row positions by their ``(Step, Sequence)`` pair.

    Parameters
    ----------
    steps, sequences : Sequence[object]
        Per-row Step and Sequence values, positionally aligned.

    Returns
    -------
    MetadataGroups
        Row positions for each distinct ``(Step, Sequence)`` pair, in the
        order rows appear in ``steps``/``sequences``.
    """
    groups: MetadataGroups = {}
    for row_index, key in enumerate(zip(steps, sequences, strict=True)):
        groups.setdefault(key, []).append(row_index)
    return groups


def _apply_t_stage(
    times: np.ndarray,
    values: np.ndarray,
    groups: MetadataGroups,
    needed_row_positions: np.ndarray,
    t_smoothing_window: float | None,
) -> np.ndarray:
    """Compute Time-direction smoothing output only at needed row positions.

    For each needed row, averages over every row in its ``(Step, Sequence)``
    group within the closed Time window, exactly as ``apply_t_smoothing``
    does for every row; this only skips positions that are not needed.

    Parameters
    ----------
    times : np.ndarray
        Full per-row Time values (float64), before row reduction.
    values : np.ndarray
        Full ``(row, wavelength)`` raw value matrix, before row reduction.
    groups : MetadataGroups
        Row positions for each ``(Step, Sequence)`` pair, over the full row
        set.
    needed_row_positions : np.ndarray
        Sorted row positions (into ``times``/``values``) to compute output
        for: the union of rows that survive downsampling and rows required
        for a Time-normalization reference mean.
    t_smoothing_window : float | None
        Positive half-window width, or ``None`` to skip smoothing.

    Returns
    -------
    np.ndarray
        ``(needed row count, wavelength count)`` matrix: smoothed values if
        ``t_smoothing_window`` is given, otherwise the raw values at the
        needed rows.

    Raises
    ------
    ValueError
        If ``t_smoothing_window`` is not finite and greater than zero.
    """
    if t_smoothing_window is None:
        return values[needed_row_positions]

    window = validate_positive_finite(t_smoothing_window, "t_smoothing_window")
    needed_row_set = set(needed_row_positions.tolist())
    position_of_row = {
        row_index: position for position, row_index in enumerate(needed_row_positions.tolist())
    }
    output = np.empty((needed_row_positions.size, values.shape[1]), dtype=np.float64)
    for indices in groups.values():
        group_indices = np.asarray(indices)
        group_times = times[group_indices]
        group_values = values[group_indices]
        for local_index, row_index in enumerate(indices):
            if row_index not in needed_row_set:
                continue
            in_window = np.abs(group_times - group_times[local_index]) <= window
            output[position_of_row[row_index]] = group_values[in_window].mean(axis=0)
    return output


def _apply_w_stage(
    values: np.ndarray,
    wavelengths: np.ndarray,
    needed_col_positions: np.ndarray,
    w_smoothing_window: float | None,
) -> np.ndarray:
    """Compute wavelength-direction smoothing output only at needed columns.

    For each needed column, averages over every column within the closed
    wavelength window, exactly as ``apply_w_smoothing`` does for every
    column; this only skips positions that are not needed.

    Parameters
    ----------
    values : np.ndarray
        ``(row, wavelength)`` matrix after the Time-direction stage, over
        the full (native-order) wavelength-column set.
    wavelengths : np.ndarray
        Wavelength value for each column of ``values``, in the same order.
    needed_col_positions : np.ndarray
        Sorted column positions to compute output for: the union of columns
        that survive downsampling and columns required for a
        wavelength-normalization reference mean.
    w_smoothing_window : float | None
        Positive half-window width, or ``None`` to skip smoothing.

    Returns
    -------
    np.ndarray
        ``(row count, needed column count)`` matrix.

    Raises
    ------
    ValueError
        If ``w_smoothing_window`` is not finite and greater than zero.
    """
    if w_smoothing_window is None:
        return values[:, needed_col_positions]

    window = validate_positive_finite(w_smoothing_window, "w_smoothing_window")
    output = np.empty((values.shape[0], needed_col_positions.size), dtype=np.float64)
    for output_index, column_index in enumerate(needed_col_positions.tolist()):
        in_window = np.abs(wavelengths - wavelengths[column_index]) <= window
        output[:, output_index] = values[:, in_window].mean(axis=1)
    return output


def _validate_full_t_normalization_groups(
    times: np.ndarray,
    groups: MetadataGroups,
    t_normalization_range: tuple[float, float] | None,
) -> None:
    """Reject groups with no Time-reference observations, over every row.

    ``_apply_t_normalization_stage`` repeats this same check, but only over
    the rows already selected as "needed" for downsampling or the
    reference mean; a ``(Step, Sequence)`` group whose rows are all
    unneeded for both reasons would otherwise never reach that check at
    all, silently skipping a validation the deferred path always performs.
    This runs once, before any row reduction, over every row the file
    actually has, so it always sees every group.

    Parameters
    ----------
    times : np.ndarray
        Full per-row Time values (float64), before row reduction.
    groups : MetadataGroups
        Row positions for each ``(Step, Sequence)`` pair, over the full row
        set.
    t_normalization_range : tuple[float, float] | None
        Inclusive reference interval, or ``None`` to skip the check.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite, or if any group
        has no reference observations.
    """
    if t_normalization_range is None:
        return
    lower, upper = validate_ordered_range(t_normalization_range, "t_normalization_range")
    if not groups:
        raise ValueError("t_normalization_range reference interval is empty for a group")
    for indices in groups.values():
        group_times = times[np.asarray(indices)]
        if not ((group_times >= lower) & (group_times <= upper)).any():
            raise ValueError(
                "t_normalization_range reference interval is empty for a group"
            )


def _apply_t_normalization_stage(
    values: np.ndarray,
    times: np.ndarray,
    groups: MetadataGroups,
    t_normalization_range: tuple[float, float] | None,
) -> None:
    """Divide each row, in place, by its group's Time-reference mean.

    Mirrors ``apply_t_normalization`` exactly, operating on the
    already-row-reduced ``values``/``times``/``groups``, which are
    guaranteed to still contain every row needed for the reference mean.

    Parameters
    ----------
    values : np.ndarray
        ``(row, wavelength)`` matrix after the Time- and wavelength-direction
        smoothing stages; modified in place.
    times : np.ndarray
        Per-row Time values, aligned with ``values``.
    groups : MetadataGroups
        Row positions for each ``(Step, Sequence)`` pair, over the same
        (already reduced) row set as ``values``/``times``.
    t_normalization_range : tuple[float, float] | None
        Inclusive reference interval, or ``None`` to skip normalization.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; a group has no
        reference observations; or a reference mean is zero or nonfinite.
    """
    if t_normalization_range is None:
        return
    lower, upper = validate_ordered_range(t_normalization_range, "t_normalization_range")
    if not groups:
        raise ValueError("t_normalization_range reference interval is empty for a group")
    for indices in groups.values():
        group_indices = np.asarray(indices)
        in_reference = (times[group_indices] >= lower) & (times[group_indices] <= upper)
        if not in_reference.any():
            raise ValueError(
                "t_normalization_range reference interval is empty for a group"
            )
        reference_mean = values[group_indices[in_reference]].mean(axis=0)
        if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
            raise ValueError(
                "t_normalization_range reference mean must be finite and nonzero"
            )
        values[group_indices] = values[group_indices] / reference_mean


def _apply_w_normalization_stage(
    values: np.ndarray,
    wavelengths: np.ndarray,
    w_normalization_range: tuple[float, float] | None,
) -> np.ndarray:
    """Divide each row by its own wavelength-reference mean.

    Mirrors ``apply_w_normalization`` exactly, operating on the
    already-column-reduced ``values``/``wavelengths``, which are guaranteed
    to still contain every column needed for the reference mean.

    Parameters
    ----------
    values : np.ndarray
        ``(row, wavelength)`` matrix after the Time- and wavelength-direction
        smoothing stages and Time-direction normalization.
    wavelengths : np.ndarray
        Wavelength value for each column of ``values``, aligned with it.
    w_normalization_range : tuple[float, float] | None
        Inclusive reference interval, or ``None`` to skip normalization.

    Returns
    -------
    np.ndarray
        ``values`` unchanged if ``w_normalization_range`` is ``None``,
        otherwise the row-normalized matrix.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; the interval has
        no wavelengths; or a row's reference mean is zero or nonfinite.
    """
    if w_normalization_range is None:
        return values
    lower, upper = validate_ordered_range(w_normalization_range, "w_normalization_range")
    in_reference = (wavelengths >= lower) & (wavelengths <= upper)
    if not in_reference.any():
        raise ValueError("w_normalization_range reference interval is empty")
    reference_mean = values[:, in_reference].mean(axis=1)
    if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
        raise ValueError("w_normalization_range reference mean must be finite and nonzero")
    return values / reference_mean[:, np.newaxis]


def _process_file(
    combined_raw: pl.DataFrame,
    *,
    target_steps: list[int] | None,
    edge_trim: Sequence[float] | None,
    t_smoothing_window: float | None,
    w_smoothing_window: float | None,
    t_normalization_range: tuple[float, float] | None,
    w_normalization_range: tuple[float, float] | None,
    selected_times_array: np.ndarray,
    selected_wavelengths: list[float],
    wavelength_columns_native: list[str],
) -> pl.DataFrame:
    """Run the full selective preprocessing pipeline for one file.

    Parameters
    ----------
    combined_raw : pl.DataFrame
        This file's freshly read ``Time``, ``Step``, ``Sequence``, and
        wavelength-range-restricted (native order) columns, not yet
        filtered, sorted, or trimmed.
    target_steps, edge_trim : see ``preprocess_and_flatten``.
    t_smoothing_window, w_smoothing_window : see ``preprocess_and_flatten``.
    t_normalization_range, w_normalization_range : see ``preprocess_and_flatten``.
    selected_times_array : np.ndarray
        Shared downsampled Time values (float64), sorted ascending.
    selected_wavelengths : list[float]
        Shared downsampled wavelength values, sorted ascending.
    wavelength_columns_native : list[str]
        This file's wavelength columns, in its own native (schema) order,
        matching ``combined_raw``'s wavelength columns.

    Returns
    -------
    pl.DataFrame
        One row per retained ``(Step, Sequence, StepTime)`` combination, with
        ``Step``, ``Sequence``, ``StepTime``, and the downsampled wavelength
        columns named canonically and sorted ascending -- ready to hand to
        ``flatten_and_prune_inputs`` alongside every other file.
    """
    filtered = filter_target_steps(combined_raw, target_steps)
    assert isinstance(filtered, pl.DataFrame)
    with_step_time = add_step_time_columns(filtered)
    assert isinstance(with_step_time, pl.DataFrame)
    trimmed = apply_edge_trim(with_step_time, edge_trim)
    assert isinstance(trimmed, pl.DataFrame)

    times = trimmed["Time"].cast(pl.Float64).to_numpy()
    steps = trimmed["Step"].to_list()
    sequences = trimmed["Sequence"].to_list()
    step_times = trimmed["StepTime"].to_numpy()
    raw_values = (
        trimmed.select(wavelength_columns_native).to_numpy().astype(np.float64, copy=False)
    )
    wavelengths_native = np.asarray(
        [parse_wavelength(column) for column in wavelength_columns_native]
    )

    _validate_full_t_normalization_groups(
        times, _group_row_indices(steps, sequences), t_normalization_range
    )

    needed_row_mask = np.isin(times, selected_times_array)
    if t_normalization_range is not None:
        t_lower, t_upper = validate_ordered_range(
            t_normalization_range, "t_normalization_range"
        )
        needed_row_mask = needed_row_mask | ((times >= t_lower) & (times <= t_upper))
    needed_row_positions = np.flatnonzero(needed_row_mask)

    t_stage_values = _apply_t_stage(
        times,
        raw_values,
        _group_row_indices(steps, sequences),
        needed_row_positions,
        t_smoothing_window,
    )
    times = times[needed_row_positions]
    steps = [steps[index] for index in needed_row_positions.tolist()]
    sequences = [sequences[index] for index in needed_row_positions.tolist()]
    step_times = step_times[needed_row_positions]

    needed_col_mask = np.isin(wavelengths_native, np.asarray(selected_wavelengths))
    if w_normalization_range is not None:
        w_lower, w_upper = validate_ordered_range(
            w_normalization_range, "w_normalization_range"
        )
        needed_col_mask = needed_col_mask | (
            (wavelengths_native >= w_lower) & (wavelengths_native <= w_upper)
        )
    needed_col_positions = np.flatnonzero(needed_col_mask)

    values = _apply_w_stage(
        t_stage_values, wavelengths_native, needed_col_positions, w_smoothing_window
    )
    wavelengths = wavelengths_native[needed_col_positions]

    groups = _group_row_indices(steps, sequences)
    _apply_t_normalization_stage(values, times, groups, t_normalization_range)
    values = _apply_w_normalization_stage(values, wavelengths, w_normalization_range)

    retained_row_positions = np.flatnonzero(np.isin(times, selected_times_array))
    wavelength_position = {
        wavelength: position for position, wavelength in enumerate(wavelengths.tolist())
    }
    retained_col_positions = [
        wavelength_position[wavelength] for wavelength in selected_wavelengths
    ]

    final_values = values[np.ix_(retained_row_positions, retained_col_positions)]
    final_columns = [f"{wavelength:.1f}nm" for wavelength in selected_wavelengths]
    metadata_frame = pl.DataFrame(
        {
            "Step": [steps[index] for index in retained_row_positions.tolist()],
            "Sequence": [sequences[index] for index in retained_row_positions.tolist()],
            "StepTime": step_times[retained_row_positions],
        }
    )
    value_frame = pl.DataFrame(
        final_values, schema=final_columns, orient="row", nan_to_null=True
    )
    return metadata_frame.hstack(value_frame)


def build_numpy_prepared_inputs(
    paths: Sequence[Path],
    *,
    target_steps: list[int] | None,
    edge_trim: Sequence[float] | None,
    wavelength_range: tuple[float, float] | None,
    t_smoothing_window: float | None,
    w_smoothing_window: float | None,
    t_normalization_range: tuple[float, float] | None,
    w_normalization_range: tuple[float, float] | None,
    t_downsampling_stride: int,
    w_downsampling_stride: int,
    validate_metadata_uniqueness: bool,
    validate_metadata_alignment: bool,
) -> list[tuple[Path, pl.DataFrame]]:
    """Validate and preprocess Flatten-PCA inputs through the NumPy fast path.

    Used by ``preprocess_and_flatten``'s ``materialize_once=True`` branch in
    place of ``load_and_validate_inputs`` plus the per-file
    smoothing/normalization/downsampling loop. Each file is read exactly
    once for its values (after a cheap metadata-only pre-read used only to
    resolve the shared Time and wavelength grids), and smoothing and
    normalization are computed only at the row and wavelength-column
    positions later stages actually need.

    Parameters
    ----------
    paths : Sequence[Path]
        Already resolved (normalized, stem-checked) input paths, as returned
        by ``resolve_and_check_paths``.
    target_steps, edge_trim, wavelength_range : see ``preprocess_and_flatten``.
    t_smoothing_window, w_smoothing_window : see ``preprocess_and_flatten``.
    t_normalization_range, w_normalization_range : see ``preprocess_and_flatten``.
    t_downsampling_stride, w_downsampling_stride : see ``preprocess_and_flatten``.
    validate_metadata_uniqueness, validate_metadata_alignment : see ``preprocess_and_flatten``.

    Returns
    -------
    list[tuple[Path, pl.DataFrame]]
        Each input path paired with its fully preprocessed, downsampled
        per-file frame, ready for ``flatten_and_prune_inputs``.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If input data violates the schema, value, requested uniqueness, or
        cross-file consistency requirements, or if any preprocessing
        argument is invalid.
    """
    # Validate every file's schema up front, before reading any column, so a
    # structurally invalid file (missing column, bad dtype, ...) is reported
    # with the same ValueError as load_and_validate_inputs rather than a raw
    # polars error from a later, schema-assuming read.
    expected_wavelength_set: frozenset[str] | None = None
    native_wavelength_columns_by_path: dict[Path, list[str]] = {}
    for path in paths:
        schema_frame = read_parquet(path)
        wavelength_set = validate_frame_schema(path, schema_frame)
        if expected_wavelength_set is None:
            expected_wavelength_set = wavelength_set
        elif wavelength_set != expected_wavelength_set:
            raise ValueError("input wavelength sets must match")
        native_wavelength_columns_by_path[path] = wavelength_columns(
            schema_frame.collect_schema().names()
        )

    if validate_metadata_alignment:
        raw_metadata = [
            (path, _read_file_metadata(path, target_steps)) for path in paths
        ]
        _validate_metadata_alignment(raw_metadata)
        trimmed_metadata = [
            (path, _prepare_file_metadata(metadata, edge_trim))
            for path, metadata in raw_metadata
        ]
    else:
        trimmed_metadata = [
            (path, _prepare_file_metadata(_read_file_metadata(path, target_steps), edge_trim))
            for path in paths
        ]

    unique_times = collect_unique_times([metadata for _, metadata in trimmed_metadata])
    selected_times_array = np.asarray(
        resolve_downsampled_values(unique_times, t_downsampling_stride, "t_downsampling_stride"),
        dtype=np.float64,
    )

    validated_wavelength_range = (
        validate_wavelength_range(wavelength_range) if wavelength_range is not None else None
    )

    def _selected_native_columns(path: Path) -> list[str]:
        """Return one file's wavelength_range-restricted columns, native order."""
        native = native_wavelength_columns_by_path[path]
        if validated_wavelength_range is None:
            return native
        return select_wavelength_columns_in_range(native, validated_wavelength_range)

    first_selected = _selected_native_columns(paths[0])
    if validated_wavelength_range is not None and not first_selected:
        raise ValueError("wavelength_range matches no wavelength columns")
    unique_wavelengths = sorted({parse_wavelength(column) for column in first_selected})
    selected_wavelengths = resolve_downsampled_values(
        unique_wavelengths, w_downsampling_stride, "w_downsampling_stride"
    )

    value_check_columns = [*METADATA_COLUMNS, *first_selected]

    prepared: list[tuple[Path, pl.DataFrame]] = []
    for path, _ in trimmed_metadata:
        file_native_columns = _selected_native_columns(path)
        combined_raw = (
            read_parquet(path)
            .select("Time", "Step", "Sequence", *file_native_columns)
            .collect()
        )
        _validate_values(path, combined_raw, value_check_columns)
        if validate_metadata_uniqueness:
            _validate_no_duplicate_metadata(path, combined_raw)

        prepared.append((
            path,
            _process_file(
                combined_raw,
                target_steps=target_steps,
                edge_trim=edge_trim,
                t_smoothing_window=t_smoothing_window,
                w_smoothing_window=w_smoothing_window,
                t_normalization_range=t_normalization_range,
                w_normalization_range=w_normalization_range,
                selected_times_array=selected_times_array,
                selected_wavelengths=selected_wavelengths,
                wavelength_columns_native=file_native_columns,
            ),
        ))
    return prepared
