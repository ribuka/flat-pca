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

from .feature_matrix import (
    MetadataKey,
    build_feature_matrix,
    build_flattened_frame,
    select_dense_features,
)

FeatureSpec = tuple[str, int, int, float, str]


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
    Materialized inputs are flattened through a NumPy feature matrix, so
    feature columns are ``Float64`` regardless of the input spectral dtype.
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

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.LazyFrame]]
        Validated paths and spectral scan query plans.

    Returns
    -------
    pl.LazyFrame
        Deferred flattened rows ordered by normalized input path.

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

    rows = []
    for path, frame in sorted(inputs, key=lambda item: str(item[0].resolve())):
        rows.append(
            frame.select(
                pl.lit(path.as_posix()).alias(SOURCE_COLUMN),
                *[
                    pl.col(column)
                    .filter(
                        (pl.col("Step") == step)
                        & (pl.col("Sequence") == sequence)
                        & (pl.col("StepTime") == step_time)
                    )
                    .first()
                    .alias(name)
                    for column, step, sequence, step_time, name in layout.feature_specs
                ],
            )
        )
    return pl.concat(rows).select(SOURCE_COLUMN, *layout.feature_names)
