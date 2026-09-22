"""Public API orchestration for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ..pca import ImputeStrategy, PcaModel
from ..preprocess import (
    add_step_time_columns,
    apply_edge_trim,
    apply_t_downsampling,
    apply_t_normalization,
    apply_t_smoothing,
    apply_w_downsampling,
    apply_w_normalization,
    apply_w_smoothing,
    apply_wavelength_range_filter,
    collect_unique_times,
    collect_unique_wavelengths,
    drop_sparse_feature_columns,
    filter_target_steps,
    validate_max_null_ratio,
)
from .flatten import flatten_inputs
from .input import StemUniquenessCheck, load_and_validate_inputs
from .input import (
    validate_metadata_alignment as validate_alignment_across_inputs,
)
from .pca_scores import fit_flattened_pca


def flatten_pca(
    paths: Sequence[str | Path] | None = None,
    *,
    flattened: pl.LazyFrame | None = None,
    n_component: int | None = None,
    target_steps: list[int] | None = None,
    edge_trim: Sequence[float] | None = None,
    wavelength_range: tuple[float, float] | None = None,
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
    impute_strategy: ImputeStrategy = "drop",
    impute_kmeans_n_clusters: int | None = None,
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
    edge_trim : Sequence[float] | None, default None
        Two finite StepTime/ReverseStepTime trim thresholds, as a ``tuple``
        or ``list``, or ``None`` to disable trimming. Used only when
        ``paths`` is specified; ignored when ``flattened`` is specified. See
        ``apply_edge_trim`` for details.
    wavelength_range : tuple[float, float] | None, default None
        Inclusive wavelength interval of columns to keep, or ``None`` to keep
        every wavelength column. Used only when ``paths`` is specified;
        ignored when ``flattened`` is specified.
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
    impute_strategy : {"drop", "median", "kmeans"}, default "drop"
        Missing-value handling forwarded to PCA fitting for any nulls that
        remain after ``max_null_ratio`` pruning. ``"drop"`` discards rows
        with any remaining null; ``"median"`` imputes with each column's
        median instead; ``"kmeans"`` imputes from the nearest cluster
        centroid fitted on rows with no missing values. Applies whether
        ``paths`` or ``flattened`` is specified.
    impute_kmeans_n_clusters : int | None, default None
        Requested cluster count for ``impute_strategy="kmeans"``, forwarded
        to PCA fitting. Must be ``None`` for any other strategy.

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
            wavelength_range=wavelength_range,
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
    return fit_flattened_pca(
        flattened,
        n_component,
        impute_strategy=impute_strategy,
        impute_kmeans_n_clusters=impute_kmeans_n_clusters,
    )


def preprocess_and_flatten(
    paths: Sequence[str | Path],
    *,
    target_steps: list[int] | None = None,
    edge_trim: Sequence[float] | None = None,
    wavelength_range: tuple[float, float] | None = None,
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
    edge_trim : Sequence[float] | None, default None
        Two finite StepTime/ReverseStepTime trim thresholds, as a ``tuple``
        or ``list``, or ``None`` to disable trimming. Applied immediately
        after StepTime/ReverseStepTime generation. See ``apply_edge_trim``
        for details.
    wavelength_range : tuple[float, float] | None, default None
        Inclusive ``(lower, upper)`` wavelength interval of columns to keep,
        or ``None`` to keep every wavelength column. Applied immediately
        after StepTime/ReverseStepTime generation and before ``edge_trim``,
        so every later stage (edge trimming, smoothing, normalization,
        downsampling, and Unique-array collection) sees only the retained
        wavelength columns. Raises ``ValueError`` if the range is malformed,
        reversed, nonfinite, or matches no wavelength column.
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
        raising ``ValueError`` on mismatch. Applied after ``target_steps``
        filtering, so files may freely differ outside the retained steps.
        Skipped by default; see the Notes section for why.
    materialize_once : bool, default True
        If ``True``, collect the flatten query exactly once and return the
        result as a ``LazyFrame`` backed by that in-memory ``DataFrame``, so
        reusing the return value (e.g. for PCA fitting and score appending)
        does not re-run the Parquet read and flatten steps. Trades memory for
        avoiding repeated computation. If ``False``, return the deferred,
        not-yet-executed flatten query as before, useful when full lazy
        execution is required. Either value returns a ``pl.LazyFrame``.
        Note that ``t_normalization_range`` and ``w_normalization_range``
        each collect every input file once while validating their reference
        statistics, so specifying either one reads the input files and runs
        the preprocessing up to normalization during this call even when
        ``materialize_once`` is ``False``; only the stages from flatten
        onward stay deferred.

    Returns
    -------
    pl.LazyFrame
        Deferred one-row-per-file flattened features with ``source`` first.

    Notes
    -----
    ``validate_metadata_alignment`` is disabled by default because differing
    ``(Step, Sequence, StepTime)`` coverage across input files -- for
    example, runs with different measurement-point counts -- is an expected
    case rather than an error. The flattening stage builds its feature set
    from the union of combinations across all input files, so a file lacking
    a particular combination simply contributes ``null`` for the
    corresponding feature (see ``flatten_inputs``). Columns that are mostly
    missing across inputs are then pruned by ``max_null_ratio`` before this
    function returns, and any remaining nulls are handled by downstream PCA
    fitting's ``impute_strategy``. Enable the check only when inputs are
    required to share an identical grid.
    """
    loaded_inputs = load_and_validate_inputs(
        paths,
        stem_uniqueness=stem_uniqueness,
        validate_metadata_uniqueness=validate_metadata_uniqueness,
        wavelength_range=wavelength_range,
    )
    filtered_inputs: list[tuple[Path, pl.LazyFrame]] = [
        (path, filter_target_steps(frame, target_steps))
        for path, frame in loaded_inputs
    ]
    if validate_metadata_alignment:
        validate_alignment_across_inputs(filtered_inputs)

    step_time_inputs: list[tuple[Path, pl.LazyFrame]] = []
    for path, frame in filtered_inputs:
        with_step_time = add_step_time_columns(frame)
        wavelength_filtered = apply_wavelength_range_filter(
            with_step_time, wavelength_range
        )
        trimmed = apply_edge_trim(wavelength_filtered, edge_trim)
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

    if materialize_once:
        # Fail fast before collecting inputs and running the expensive
        # flatten below; drop_sparse_feature_columns validates again for its
        # independent callers.
        validate_max_null_ratio(max_null_ratio)
        # Collecting each file before flatten routes flatten_inputs through
        # its eager DataFrame branch (dict-based per-combo lookups), which is
        # far faster than its LazyFrame branch (one filter+first expression
        # per wavelength-and-combo cell). materialize_once=True already
        # collects the flatten result immediately afterward, so this does
        # not add a new collection boundary, only moves it earlier.
        collected_inputs: list[tuple[Path, pl.DataFrame]] = [
            (path, frame.collect()) for path, frame in prepared_inputs
        ]
        flattened = flatten_inputs(collected_inputs)
        assert isinstance(flattened, pl.DataFrame)
        return materialize_and_drop_sparse_feature_columns(
            flattened,
            max_null_ratio,
        )
    flattened = flatten_inputs(prepared_inputs)
    assert isinstance(flattened, pl.LazyFrame)
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
    flattened: pl.DataFrame | pl.LazyFrame,
    max_null_ratio: float,
) -> pl.LazyFrame:
    """Materialize once, prune sparse columns, and cache the pruned result.

    Parameters
    ----------
    flattened : pl.DataFrame | pl.LazyFrame
        Flattened features before sparse-column pruning. Already-materialized
        input (as produced by ``flatten_inputs``'s eager DataFrame branch) is
        used directly without a redundant ``.lazy().collect()`` round trip.
    max_null_ratio : float
        Inclusive maximum missing-value ratio for retained feature columns.

    Returns
    -------
    pl.LazyFrame
        An in-memory-backed LazyFrame containing the pruned flattened features.

    Notes
    -----
    Sparse-column statistics are calculated from the already materialized
    result, and the selected columns are rewrapped directly without a second
    ``collect()``. This prevents an upstream re-execution and retains only
    the pruned feature set after this function returns.
    """
    # Fail fast before collecting an unmaterialized upstream flatten query.
    # The pruning stage validates again for its independent callers.
    validate_max_null_ratio(max_null_ratio)
    materialized = flattened.collect() if isinstance(flattened, pl.LazyFrame) else flattened
    pruned = drop_sparse_feature_columns(materialized, max_null_ratio)
    assert isinstance(pruned, pl.DataFrame)
    cached = pruned.lazy()
    del materialized
    return cached
