"""Public API orchestration for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from ..pca import PcaModel
from ..preprocess import (
    add_step_time_columns,
    apply_edge_trim,
    apply_t_downsampling,
    apply_t_normalization,
    apply_t_smoothing,
    apply_w_downsampling,
    apply_w_normalization,
    apply_w_smoothing,
    collect_unique_times,
    collect_unique_wavelengths,
    drop_sparse_feature_columns,
    filter_target_steps,
    validate_max_null_ratio,
)
from .flatten import flatten_inputs
from .input import StemUniquenessCheck, load_and_validate_inputs
from .input import validate_metadata_alignment as _validate_metadata_alignment
from .pca_scores import fit_flattened_pca


def flatten_pca(
    paths: Sequence[str | Path] | None = None,
    *,
    flattened: pl.LazyFrame | None = None,
    n_component: int | None = None,
    target_steps: list[int] | None = None,
    edge_trim: list[float, float] | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
    max_null_ratio: float = 0.1,
    stem_uniqueness: StemUniquenessCheck = "skip",
    validate_metadata_uniqueness: bool = False,
    validate_metadata_alignment: bool = False,
    materialize_once: bool = True,
    impute_strategy: Literal["drop", "median"] = "drop",
) -> PcaModel:
    """Preprocess, flatten, and fit PCA across spectral Parquet files.

    Parameters
    ----------
    paths : Sequence[str | Path] | None, default None
        One or more Parquet input paths. Exactly one of ``paths`` and
        ``flattened`` must be supplied.
    flattened : pl.LazyFrame | None, default None
        Existing flattened features used directly for PCA fitting.
    n_component : int | None, default None
        Number of PCA score columns to append. If ``None``, use the maximum
        available count: the smaller of the input-file count and flattened
        spectral-feature count.
    target_steps : list[int] | None, default None
        ``Step`` values to keep, or ``None`` to keep every row. Used only
        when ``paths`` is specified; ignored when ``flattened`` is specified.
    edge_trim : list[float, float] | None, default None
        ``(edge_trim[0], edge_trim[1])`` StepTime/ReverseStepTime trim
        thresholds, or ``None`` to disable trimming. Used only when ``paths``
        is specified; ignored when ``flattened`` is specified.
    t_smoothing_window : float | None, default None
        Positive time-direction smoothing half-window, or ``None``.
    w_smoothing_window : float | None, default None
        Positive wavelength-direction smoothing half-window, or ``None``.
    t_normalization_range : tuple[float, float] | None, default None
        Inclusive time interval used for normalization, or ``None``.
    w_normalization_range : tuple[float, float] | None, default None
        Inclusive wavelength interval used for normalization, or ``None``.
    t_downsampling_stride : int, default 1
        Interval between retained values in the shared sorted Time array.
    w_downsampling_stride : int, default 1
        Interval between retained values in the shared sorted wavelength array.
    max_null_ratio : float, default 0.1
        Forwarded to ``preprocess_and_flatten`` when ``paths`` is specified;
        ignored when ``flattened`` is specified. See
        ``preprocess_and_flatten`` for details.
    stem_uniqueness : Literal["skip", "warn", "error"], default "skip"
        How to handle duplicate ``Path.stem`` values across ``paths``. Used
        only when ``paths`` is specified; ignored when ``flattened`` is
        specified. See ``load_and_validate_inputs`` for details.
    validate_metadata_uniqueness : bool, default False
        Forwarded to ``preprocess_and_flatten`` when ``paths`` is specified;
        ignored when ``flattened`` is specified. See
        ``preprocess_and_flatten`` for details.
    validate_metadata_alignment : bool, default False
        Forwarded to ``preprocess_and_flatten`` when ``paths`` is specified;
        ignored when ``flattened`` is specified. See
        ``preprocess_and_flatten`` for details.
    materialize_once : bool, default True
        Forwarded to ``preprocess_and_flatten`` when ``paths`` is specified;
        ignored when ``flattened`` is specified. See
        ``preprocess_and_flatten`` for details.
    impute_strategy : {"drop", "median"}, default "drop"
        Missing-value handling forwarded to PCA fitting for any nulls that
        remain after ``max_null_ratio`` pruning. ``"drop"`` discards rows
        with any remaining null; ``"median"`` imputes with each column's
        median instead. Applies whether ``paths`` or ``flattened`` is
        specified.

    Returns
    -------
    PcaModel
        Fitted PCA pipeline state.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If inputs, preprocessing arguments, or ``n_component`` are invalid.
    """
    if (paths is None) == (flattened is None):
        raise ValueError("exactly one of paths or flattened must be specified")
    if flattened is None:
        assert paths is not None
        flattened = preprocess_and_flatten(
            paths,
            target_steps=target_steps,
            edge_trim=edge_trim,
            t_smoothing_window=t_smoothing_window,
            w_smoothing_window=w_smoothing_window,
            t_normalization_range=t_normalization_range,
            w_normalization_range=w_normalization_range,
            t_downsampling_stride=t_downsampling_stride,
            w_downsampling_stride=w_downsampling_stride,
            max_null_ratio=max_null_ratio,
            stem_uniqueness=stem_uniqueness,
            validate_metadata_uniqueness=validate_metadata_uniqueness,
            validate_metadata_alignment=validate_metadata_alignment,
            materialize_once=materialize_once,
        )
    return fit_flattened_pca(flattened, n_component, impute_strategy=impute_strategy)


def preprocess_and_flatten(
    paths: Sequence[str | Path],
    *,
    target_steps: list[int] | None = None,
    edge_trim: list[float, float] | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
    max_null_ratio: float = 0.1,
    stem_uniqueness: StemUniquenessCheck = "skip",
    validate_metadata_uniqueness: bool = False,
    validate_metadata_alignment: bool = False,
    materialize_once: bool = True,
) -> pl.LazyFrame:
    """Build a lazy preprocessing and deterministic flattening query.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more Parquet input paths.
    target_steps : list[int] | None, default None
        ``Step`` values to keep, or ``None`` to keep every row. Applied after
        input validation and before StepTime/ReverseStepTime generation.
    edge_trim : list[float, float] | None, default None
        ``(edge_trim[0], edge_trim[1])`` StepTime/ReverseStepTime trim
        thresholds, or ``None`` to disable trimming. Applied immediately
        after StepTime/ReverseStepTime generation.
    t_smoothing_window, w_smoothing_window : float | None, default None
        Positive smoothing half-windows, or ``None`` to disable each stage.
    t_normalization_range, w_normalization_range : tuple[float, float] | None, default None
        Inclusive normalization ranges, or ``None`` to disable each stage.
    t_downsampling_stride, w_downsampling_stride : int, default 1
        Positive intervals in the shared sorted Time and wavelength arrays.
    max_null_ratio : float, default 0.1
        Upper bound (inclusive) on a flattened feature column's null-or-NaN
        ratio for it to be kept; sparser columns are dropped. When
        ``materialize_once=True``, flatten results are materialized before
        pruning and the pruned result is then cached. When ``False``, the
        existing deferred prune query is returned. In either case,
        ``flatten_pca``, ``append_pca_scores``, and
        ``reshape_pca_components`` see the same pruned column set. Use
        ``1.0`` to disable pruning (only an entirely null/NaN column would
        still be dropped). See ``drop_sparse_feature_columns``.
    stem_uniqueness : Literal["skip", "warn", "error"], default "skip"
        How to handle duplicate ``Path.stem`` values across ``paths``. See
        ``load_and_validate_inputs`` for details.
    validate_metadata_uniqueness : bool, default False
        If ``True``, reject duplicate ``(Time, Step, Sequence)`` tuples
        within each input file. Skipped by default because input producers
        already guarantee uniqueness. Applied before preprocessing; see
        ``load_and_validate_inputs`` for details.
    validate_metadata_alignment : bool, default False
        If ``True``, require every input file to share an identical set of
        ``(Time, Step, Sequence)`` tuples after ``target_steps`` filtering,
        raising ``ValueError`` on mismatch. Skipped by default, because
        differing ``(Step, Sequence, StepTime)`` coverage across input files
        (for example, runs with different measurement-point counts) is an
        expected case, not an error: the flattening stage builds its feature
        set from the union of combinations across all input files, and a
        file lacking a particular combination simply contributes ``null``
        for that feature (see ``flatten_inputs``). Columns that are mostly
        missing across inputs are then pruned by ``max_null_ratio`` before
        this function returns, and any remaining nulls are handled by
        downstream PCA fitting's ``impute_strategy``. Enable this check only
        when you want to enforce that inputs share an identical grid.
        Applied after ``target_steps`` filtering so files may freely differ
        outside the retained steps.
    materialize_once : bool, default True
        If ``True``, collect the flatten query exactly once and return the
        result as a ``LazyFrame`` backed by that in-memory ``DataFrame``, so
        reusing the return value (e.g. for PCA fitting and score appending)
        does not re-run the Parquet read and flatten steps. Trades memory for
        avoiding repeated computation. If ``False``, return the deferred,
        not-yet-executed flatten query as before, useful when full lazy
        execution is required. Either value returns a ``pl.LazyFrame``.

    Returns
    -------
    pl.LazyFrame
        Deferred one-row-per-file flattened features with ``source`` first.
    """
    loaded_inputs = load_and_validate_inputs(
        paths,
        stem_uniqueness=stem_uniqueness,
        validate_metadata_uniqueness=validate_metadata_uniqueness,
    )
    filtered_inputs: list[tuple[Path, pl.LazyFrame]] = [
        (path, filter_target_steps(frame, target_steps))
        for path, frame in loaded_inputs
    ]
    if validate_metadata_alignment:
        _validate_metadata_alignment(filtered_inputs)

    step_time_inputs: list[tuple[Path, pl.LazyFrame]] = []
    for path, frame in filtered_inputs:
        with_step_time = add_step_time_columns(frame)
        trimmed = apply_edge_trim(with_step_time, edge_trim)
        step_time_inputs.append((path, trimmed))

    frames = [frame for _, frame in step_time_inputs]
    unique_times = collect_unique_times(frames)
    unique_wavelengths = collect_unique_wavelengths(frames)
    prepared_inputs: list[tuple[Path, pl.LazyFrame]] = []
    for path, frame in step_time_inputs:
        prepared = apply_t_smoothing(frame, t_smoothing_window)
        prepared = apply_w_smoothing(prepared, w_smoothing_window)
        prepared = apply_t_normalization(prepared, t_normalization_range)
        prepared = apply_w_normalization(prepared, w_normalization_range)
        prepared = apply_t_downsampling(
            prepared,
            unique_times,
            t_downsampling_stride,
        )
        prepared = apply_w_downsampling(
            prepared,
            unique_wavelengths,
            w_downsampling_stride,
        )
        prepared_inputs.append((path, prepared))

    flattened = flatten_inputs(prepared_inputs)
    assert isinstance(flattened, pl.LazyFrame)
    if materialize_once:
        return materialize_and_drop_sparse_feature_columns(
            flattened,
            max_null_ratio,
        )
    return drop_sparse_feature_columns(flattened, max_null_ratio)


def materialize_flattened(flattened: pl.LazyFrame) -> pl.LazyFrame:
    """Collect a flatten query once and rewrap it as an in-memory-backed LazyFrame.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Deferred flatten query to execute exactly once.

    Returns
    -------
    pl.LazyFrame
        ``LazyFrame`` wrapping the collected ``DataFrame``, so downstream
        ``.collect()`` calls reuse the materialized result instead of
        re-running the underlying query.
    """
    return flattened.collect().lazy()


def materialize_and_drop_sparse_feature_columns(
    flattened: pl.LazyFrame,
    max_null_ratio: float,
) -> pl.LazyFrame:
    """Materialize once, prune sparse columns, and cache the pruned result.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Deferred flattened features before sparse-column pruning.
    max_null_ratio : float
        Inclusive maximum missing-value ratio for retained feature columns.

    Returns
    -------
    pl.LazyFrame
        An in-memory-backed LazyFrame containing the pruned flattened features.

    Notes
    -----
    Sparse-column statistics are calculated from the already materialized
    result. This prevents the expensive upstream Parquet and flatten query
    from being run both for statistics and for the cached return value.
    """
    # Fail fast before materializing the expensive upstream flatten query.
    # The pruning stage validates again for its independent callers.
    validate_max_null_ratio(max_null_ratio)
    materialized = materialize_flattened(flattened)
    pruned = drop_sparse_feature_columns(materialized, max_null_ratio)
    return materialize_flattened(pruned)
