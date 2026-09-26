"""Deterministic spectral flattening for Flatten-PCA."""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import polars as pl

from flat_pca.spectral.schema import (
    SOURCE_COLUMN,
    parse_wavelength,
    wavelength_columns,
)

from ..preprocess.sparse_columns import drop_sparse_feature_columns
from .feature_matrix import (
    MetadataKey,
    build_feature_matrix,
    build_flattened_frame,
    select_dense_features,
)

FeatureSpec = tuple[str, int, int, float, str]
METADATA_KEY_COLUMNS = ("Step", "Sequence", "StepTime")


class FlattenLayout(NamedTuple):
    """Column layout shared by every flattening strategy.

    Attributes
    ----------
    wavelength_columns_sorted : list[str]
        Wavelength column names ordered by numeric wavelength.
    metadata_rows : list[MetadataKey]
        Sorted union of ``(Step, Sequence, StepTime)`` combinations.
    feature_specs : list[FeatureSpec]
        ``(column, step, sequence, step_time, name)`` tuples in feature order.
    feature_names : list[str]
        Formatted feature names in feature order.
    """

    wavelength_columns_sorted: list[str]
    metadata_rows: list[MetadataKey]
    feature_specs: list[FeatureSpec]
    feature_names: list[str]


def _collect_metadata_rows(
    frame: pl.DataFrame | pl.LazyFrame,
) -> list[tuple[int, int, float]]:
    """Return one frame's own unique (Step, Sequence, StepTime) rows.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Validated spectral frame to inspect.

    Returns
    -------
    list[tuple[int, int, float]]
        Unique ``(Step, Sequence, StepTime)`` rows present in ``frame``.
    """
    selected = frame.select("Step", "Sequence", "StepTime").unique()
    collected = selected.collect() if isinstance(selected, pl.LazyFrame) else selected
    return list(collected.iter_rows())


def _union_metadata_rows(
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
) -> list[MetadataKey]:
    """Union the Step/Sequence/StepTime combinations across every frame.

    Different input files may cover different ``(Step, Sequence, StepTime)``
    combinations (for example, runs with different measurement-point
    counts), so the feature set is the union across all frames rather than
    any single frame's own combinations.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame | pl.LazyFrame]
        Validated spectral frames to union.

    Returns
    -------
    list[MetadataKey]
        Unique ``(Step, Sequence, StepTime)`` combinations in sorted order.
    """
    metadata_keys: set[MetadataKey] = set()
    for frame in frames:
        metadata_keys.update(_collect_metadata_rows(frame))
    return sorted(metadata_keys)


def _build_feature_specs(
    wavelength_columns_sorted: list[str],
    metadata_rows: list[MetadataKey],
) -> list[FeatureSpec]:
    """Build feature specs for every wavelength and metadata combination.

    Parameters
    ----------
    wavelength_columns_sorted : list[str]
        Wavelength column names ordered by numeric wavelength.
    metadata_rows : list[MetadataKey]
        Sorted union of ``(Step, Sequence, StepTime)`` combinations.

    Returns
    -------
    list[FeatureSpec]
        ``(column, step, sequence, step_time, name)`` tuples ordered by
        numeric wavelength, then Step, Sequence, and StepTime.
    """
    return [
        (
            column,
            step,
            sequence,
            step_time,
            f"{column}_{int(step)}_{int(sequence)}_{float(step_time):.2f}",
        )
        for column in wavelength_columns_sorted
        for step, sequence, step_time in metadata_rows
    ]


def _sorted_wavelength_columns(columns: list[str]) -> list[str]:
    """Order a frame's wavelength columns by their numeric wavelength.

    Parameters
    ----------
    columns : list[str]
        Complete column names of one validated spectral frame.

    Returns
    -------
    list[str]
        Wavelength column names in ascending numeric order.
    """
    return [
        column
        for _, column in sorted(
            (
                (parse_wavelength(column), column)
                for column in wavelength_columns(columns)
            ),
            key=lambda item: item[0],
        )
    ]


def _prepare_flatten_layout(
    columns: list[str],
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
) -> FlattenLayout:
    """Resolve the column layout shared by both flattening strategies.

    Parameters
    ----------
    columns : list[str]
        Complete column names of the first input frame, which every input
        shares by validation.
    frames : Sequence[pl.DataFrame | pl.LazyFrame]
        Validated spectral frames whose metadata combinations are unioned.

    Returns
    -------
    FlattenLayout
        Sorted wavelength columns, metadata combinations, feature specs, and
        feature names.

    Raises
    ------
    ValueError
        If formatted feature names collide.
    """
    wavelength_columns_sorted = _sorted_wavelength_columns(columns)
    metadata_rows = _union_metadata_rows(frames)
    feature_specs = _build_feature_specs(wavelength_columns_sorted, metadata_rows)
    feature_names = [spec[-1] for spec in feature_specs]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate flattened feature names")
    return FlattenLayout(
        wavelength_columns_sorted, metadata_rows, feature_specs, feature_names
    )


def flatten_inputs(
    inputs: Sequence[tuple[Path, pl.DataFrame | pl.LazyFrame]],
) -> pl.DataFrame | pl.LazyFrame:
    """Flatten each validated spectral frame into one deterministic row.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame | pl.LazyFrame]]
        Normalized input paths paired with validated spectral frames.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        One row per input file, ordered by normalized path, with ``source``
        followed by spectral features ordered by numeric wavelength, Step,
        Sequence, and StepTime. Features correspond to the union of
        ``(Step, Sequence, StepTime)`` combinations across all input frames;
        a frame lacking a particular combination contributes ``null`` for
        the corresponding feature.

    Raises
    ------
    ValueError
        If no inputs are provided, input frames mix ``pl.DataFrame`` and
        ``pl.LazyFrame`` types, or formatted feature names collide.

    Notes
    -----
    Materialized inputs whose wavelength columns are all floating point are
    flattened through a ``float64`` NumPy feature matrix, which widens
    ``Float32`` to ``Float64`` exactly as the row-dict flattening did. Any
    other numeric wavelength dtype (integer or decimal) keeps the row-dict
    path, so its flattened dtype and values are unchanged.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must contain at least one validated frame")

    if any(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
        if not all(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
            raise ValueError("flatten inputs must use one frame type")
        return _flatten_lazy_inputs(inputs)  # type: ignore[arg-type]

    sorted_inputs = _sort_eager_inputs(inputs)  # type: ignore[arg-type]
    layout = _prepare_flatten_layout(
        sorted_inputs[0][1].columns,
        [frame for _, frame in sorted_inputs],
    )
    if not _has_float_spectral_columns(sorted_inputs, layout):
        return _flatten_rows_eager(sorted_inputs, layout)
    matrix = build_feature_matrix(
        [frame for _, frame in sorted_inputs],
        layout.wavelength_columns_sorted,
        layout.metadata_rows,
    )
    return build_flattened_frame(
        matrix,
        [path.as_posix() for path, _ in sorted_inputs],
        layout.feature_names,
    )


def flatten_and_prune_inputs(
    inputs: Sequence[tuple[Path, pl.DataFrame]],
    max_null_ratio: float,
) -> pl.DataFrame:
    """Flatten materialized inputs and keep only dense feature columns.

    Flattening and sparse-column pruning are fused so that the missing-value
    ratios are measured on the NumPy feature matrix and only the retained
    columns are ever handed to polars. Building the full frame first and
    pruning it afterwards would materialize every sparse column for nothing,
    which dominates the cost when the feature count reaches six figures.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Normalized input paths paired with validated, materialized spectral
        frames.
    max_null_ratio : float
        Inclusive upper bound (0.0-1.0) on a feature column's missing-value
        ratio for it to be kept. Use ``1.0`` to keep every column except
        entirely missing ones.

    Returns
    -------
    pl.DataFrame
        One row per input file, ordered by normalized path, with ``source``
        followed by the retained spectral features. Column order matches
        ``flatten_inputs``; missing combinations are polars nulls.

    See Also
    --------
    flatten_inputs : Flattening without pruning, including the dtype rules
        this function shares. Non-floating wavelength dtypes take the
        row-dict path and are pruned afterwards by
        ``drop_sparse_feature_columns``.

    Raises
    ------
    ValueError
        If no inputs are provided, any frame is a ``pl.LazyFrame``,
        formatted feature names collide, or ``max_null_ratio`` is not
        between 0.0 and 1.0.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must contain at least one validated frame")
    if any(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
        raise ValueError("flatten inputs must be materialized frames")

    sorted_inputs = _sort_eager_inputs(inputs)
    layout = _prepare_flatten_layout(
        sorted_inputs[0][1].columns,
        [frame for _, frame in sorted_inputs],
    )
    if not _has_float_spectral_columns(sorted_inputs, layout):
        pruned = drop_sparse_feature_columns(
            _flatten_rows_eager(sorted_inputs, layout), max_null_ratio
        )
        assert isinstance(pruned, pl.DataFrame)
        return pruned
    matrix = build_feature_matrix(
        [frame for _, frame in sorted_inputs],
        layout.wavelength_columns_sorted,
        layout.metadata_rows,
    )
    dense_matrix, dense_names = select_dense_features(
        matrix, layout.feature_names, max_null_ratio
    )
    del matrix
    return build_flattened_frame(
        dense_matrix,
        [path.as_posix() for path, _ in sorted_inputs],
        dense_names,
    )


def _has_float_spectral_columns(
    inputs: Sequence[tuple[Path, pl.DataFrame]],
    layout: FlattenLayout,
) -> bool:
    """Report whether every input's wavelength columns are floating point.

    The NumPy feature matrix is ``float64``, which reproduces the row-dict
    flattening exactly for ``Float32`` and ``Float64`` inputs but not for
    integer or decimal ones: those would change the flattened dtype and, for
    integers beyond 53 bits, the values themselves.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Input paths paired with materialized spectral frames.
    layout : FlattenLayout
        Resolved column layout naming the wavelength columns to inspect.

    Returns
    -------
    bool
        ``True`` when every wavelength column of every input frame has a
        floating-point dtype.
    """
    for _, frame in inputs:
        schema = frame.schema
        if any(
            not schema[column].is_float()
            for column in layout.wavelength_columns_sorted
        ):
            return False
    return True


def _flatten_rows_eager(
    inputs: Sequence[tuple[Path, pl.DataFrame]],
    layout: FlattenLayout,
) -> pl.DataFrame:
    """Flatten materialized inputs one Python row dict at a time.

    Kept as the fallback for wavelength columns the ``float64`` feature
    matrix cannot represent losslessly, so those inputs keep the dtype and
    the exact values they had before the NumPy path existed.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Input paths paired with materialized spectral frames, already sorted
        by normalized path.
    layout : FlattenLayout
        Resolved column layout for the flattened features.

    Returns
    -------
    pl.DataFrame
        One row per input, with ``source`` followed by the features. A
        combination an input lacks contributes ``None``; a repeated
        combination keeps the frame's last row.
    """
    rows: list[dict[str, object]] = []
    for path, frame in inputs:
        by_key = {
            (row["Step"], row["Sequence"], row["StepTime"]): row
            for row in frame.select(
                "Step", "Sequence", "StepTime", *layout.wavelength_columns_sorted
            ).iter_rows(named=True)
        }
        row: dict[str, object] = {SOURCE_COLUMN: path.as_posix()}
        for column, step, sequence, step_time, name in layout.feature_specs:
            match = by_key.get((step, sequence, step_time))
            row[name] = match[column] if match is not None else None
        rows.append(row)
    return pl.DataFrame(rows).select(SOURCE_COLUMN, *layout.feature_names)


def _sort_eager_inputs(
    inputs: Sequence[tuple[Path, pl.DataFrame]],
) -> list[tuple[Path, pl.DataFrame]]:
    """Order materialized inputs by normalized path text.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Input paths paired with materialized spectral frames.

    Returns
    -------
    list[tuple[Path, pl.DataFrame]]
        ``inputs`` sorted by resolved path text, matching the deferred
        flattening strategy's row order.
    """
    return sorted(inputs, key=lambda item: str(item[0].resolve()))


def _flatten_lazy_inputs(
    inputs: Sequence[tuple[Path, pl.LazyFrame]],
) -> pl.LazyFrame:
    """Build a deferred deterministic one-row-per-input flatten query.

    Each input is aligned to the shared ``(Step, Sequence, StepTime)`` grid
    by one left join and stacked into long columns, one per flattened dtype,
    and the stacked columns are widened once per dtype at collect time. The
    query therefore grows with the input count, not with the feature count,
    which reaches six figures in practice.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.LazyFrame]]
        Validated paths and spectral scan query plans.

    Returns
    -------
    pl.LazyFrame
        Deferred flattened rows ordered by normalized input path. Values and
        dtypes match the materialized ``flatten_inputs`` branch (see
        ``_flattened_dtype``), ``NaN`` becomes null when every wavelength
        column is floating point, and a repeated combination keeps the
        frame's last row.

    Raises
    ------
    ValueError
        If formatted feature names collide.
    """
    frames = [frame for _, frame in inputs]
    layout = _prepare_flatten_layout(
        frames[0].collect_schema().names(),
        frames,
    )
    sorted_inputs = sorted(inputs, key=lambda item: str(item[0].resolve()))
    sorted_frames = [frame for _, frame in sorted_inputs]
    sources = pl.LazyFrame(
        {SOURCE_COLUMN: [path.as_posix() for path, _ in sorted_inputs]},
        schema={SOURCE_COLUMN: pl.String},
    )
    if not layout.feature_names:
        return sources

    schemas = [frame.collect_schema() for frame in frames]
    as_float64 = all(
        schema[column].is_float()
        for schema in schemas
        for column in layout.wavelength_columns_sorted
    )
    common_schema = _common_schema(
        schemas, [*METADATA_KEY_COLUMNS, *layout.wavelength_columns_sorted]
    )
    grid = _metadata_grid(layout.metadata_rows, common_schema)
    widened = [
        _widen_dtype_group(sorted_frames, grid, layout, columns, dtype, as_float64)
        for dtype, columns in _group_columns_by_flattened_dtype(
            common_schema, layout.wavelength_columns_sorted, as_float64
        ).items()
    ]
    flattened = pl.concat([sources, *widened], how="horizontal", strict=True)
    if len(widened) == 1:
        # A single group is already in feature order; reselecting six-figure
        # column counts is not free.
        return flattened
    return flattened.select(SOURCE_COLUMN, *layout.feature_names)


def _common_schema(schemas: Sequence[pl.Schema], columns: list[str]) -> pl.Schema:
    """Resolve each column's common supertype across every input schema.

    Inputs may store the same column with different numeric dtypes, so
    casting every input to one input's dtype could narrow values (for
    example ``Float64`` to ``Int64``) or make distinct metadata keys equal.

    Parameters
    ----------
    schemas : Sequence[pl.Schema]
        Schemas of every input frame.
    columns : list[str]
        Columns to resolve.

    Returns
    -------
    pl.Schema
        Supertype of each column in ``columns``, as polars resolves it when
        vertically concatenating the inputs.
    """
    return pl.concat(
        [
            pl.LazyFrame(schema={column: schema[column] for column in columns})
            for schema in schemas
        ],
        how="vertical_relaxed",
    ).collect_schema()


def _flattened_dtype(dtype: pl.DataType, as_float64: bool) -> pl.DataType:
    """Return the flattened dtype the materialized branch gives a wavelength column.

    Parameters
    ----------
    dtype : pl.DataType
        Input dtype of the wavelength column.
    as_float64 : bool
        Whether every input's wavelength columns are floating point, so the
        materialized branch uses the ``float64`` NumPy feature matrix.

    Returns
    -------
    pl.DataType
        ``Float64`` for the NumPy feature matrix and for floating columns on
        the row-dict path, ``Int64`` for signed integers and unsigned
        integers up to 32 bits (as polars infers from Python ints), and
        ``dtype`` itself otherwise.
    """
    if as_float64 or dtype.is_float():
        return pl.Float64()
    if dtype.is_signed_integer() or dtype in (pl.UInt8, pl.UInt16, pl.UInt32):
        return pl.Int64()
    return dtype


def _group_columns_by_flattened_dtype(
    schema: pl.Schema,
    wavelength_columns_sorted: list[str],
    as_float64: bool,
) -> dict[pl.DataType, list[str]]:
    """Group wavelength columns by the dtype their flattened features take.

    ``DataFrame.transpose`` needs one dtype, so each group is widened on its
    own; stacking every column into one supertype would, for example, round
    ``Int64`` values beyond 53 bits when a ``Float64`` column is present.

    Parameters
    ----------
    schema : pl.Schema
        Common schema of every input, supplying the wavelength column
        dtypes.
    wavelength_columns_sorted : list[str]
        Wavelength column names ordered by numeric wavelength.
    as_float64 : bool
        Whether every input's wavelength columns are floating point.

    Returns
    -------
    dict[pl.DataType, list[str]]
        Wavelength columns per flattened dtype, each list in wavelength
        order.
    """
    groups: dict[pl.DataType, list[str]] = {}
    for column in wavelength_columns_sorted:
        groups.setdefault(_flattened_dtype(schema[column], as_float64), []).append(
            column
        )
    return groups


def _widen_dtype_group(
    frames: Sequence[pl.LazyFrame],
    grid: pl.LazyFrame,
    layout: FlattenLayout,
    columns: list[str],
    dtype: pl.DataType,
    as_float64: bool,
) -> pl.LazyFrame:
    """Build the deferred flattened features of one dtype group.

    Parameters
    ----------
    frames : Sequence[pl.LazyFrame]
        Validated spectral query plans, ordered by normalized input path.
    grid : pl.LazyFrame
        Shared metadata grid from ``_metadata_grid``.
    layout : FlattenLayout
        Resolved column layout for the flattened features.
    columns : list[str]
        Wavelength columns of the group, in wavelength order.
    dtype : pl.DataType
        Flattened dtype of the group.
    as_float64 : bool
        Whether ``NaN`` becomes null, as in the NumPy feature matrix.

    Returns
    -------
    pl.LazyFrame
        One row per frame holding the group's features in flatten order.
    """
    group = set(columns)
    feature_names = [
        spec[-1] for spec in layout.feature_specs if spec[0] in group
    ]
    stacked = pl.concat(
        [
            _stack_aligned_features(frame, grid, columns, f"c{index}", dtype, as_float64)
            for index, frame in enumerate(frames)
        ],
        how="horizontal",
        strict=True,
    )
    return stacked.map_batches(
        lambda batch: batch.transpose(column_names=feature_names),
        schema=dict.fromkeys(feature_names, dtype),
    )


def _metadata_grid(
    metadata_rows: list[MetadataKey],
    schema: pl.Schema,
) -> pl.LazyFrame:
    """Build the shared ``(Step, Sequence, StepTime)`` grid as a LazyFrame.

    Parameters
    ----------
    metadata_rows : list[MetadataKey]
        Sorted union of ``(Step, Sequence, StepTime)`` combinations.
    schema : pl.Schema
        Common schema of every input, supplying the key column dtypes.

    Returns
    -------
    pl.LazyFrame
        One row per combination, in ``metadata_rows`` order.
    """
    return pl.LazyFrame(
        metadata_rows,
        schema={key: schema[key] for key in METADATA_KEY_COLUMNS},
        orient="row",
    )


def _stack_aligned_features(
    frame: pl.LazyFrame,
    grid: pl.LazyFrame,
    columns: list[str],
    name: str,
    dtype: pl.DataType,
    null_nan: bool,
) -> pl.LazyFrame:
    """Align one input to the metadata grid and stack it in feature order.

    Parameters
    ----------
    frame : pl.LazyFrame
        Validated spectral query plan of one input.
    grid : pl.LazyFrame
        Shared metadata grid from ``_metadata_grid``.
    columns : list[str]
        Wavelength columns to stack, in wavelength order.
    name : str
        Name of the returned column.
    dtype : pl.DataType
        Dtype of the returned column.
    null_nan : bool
        Whether ``NaN`` becomes null.

    Returns
    -------
    pl.LazyFrame
        One column of ``len(columns) * combination count`` values, ordered
        by wavelength, then by grid row, which is the flattened feature
        order. Combinations the input lacks are null, and a repeated
        combination keeps the input's last row.
    """
    keys = list(METADATA_KEY_COLUMNS)
    grid_schema = grid.collect_schema()
    deduplicated = frame.select(
        *[pl.col(key).cast(grid_schema[key]) for key in keys],
        *[pl.col(column).cast(dtype) for column in columns],
    ).unique(subset=keys, keep="last")
    aligned = grid.join(
        deduplicated, on=keys, how="left", maintain_order="left"
    ).select(columns)
    value = pl.col("value")
    if null_nan:
        value = value.fill_nan(None)
    return aligned.unpivot(on=columns).select(value.alias(name))
