"""Deterministic spectral flattening for Flatten-PCA."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from flat_pca.spectral.schema import (
    SOURCE_COLUMN,
    parse_wavelength,
    wavelength_columns,
)

FeatureSpec = tuple[str, int, int, float, str]


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


def _build_feature_specs(
    wavelength_columns_sorted: list[str],
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
) -> list[FeatureSpec]:
    """Build feature specs from the union of Step/Sequence/StepTime combinations.

    Different input files may cover different ``(Step, Sequence, StepTime)``
    combinations (for example, runs with different measurement-point
    counts), so the feature set is the union across all frames rather than
    any single frame's own combinations.

    Parameters
    ----------
    wavelength_columns_sorted : list[str]
        Wavelength column names ordered by numeric wavelength.
    frames : Sequence[pl.DataFrame | pl.LazyFrame]
        Validated spectral frames to union.

    Returns
    -------
    list[FeatureSpec]
        ``(column, step, sequence, step_time, name)`` tuples ordered by
        numeric wavelength, then Step, Sequence, and StepTime.
    """
    metadata_keys: set[tuple[int, int, float]] = set()
    for frame in frames:
        metadata_keys.update(_collect_metadata_rows(frame))
    metadata_rows = sorted(metadata_keys)
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
) -> tuple[list[str], list[FeatureSpec], list[str]]:
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
    tuple[list[str], list[FeatureSpec], list[str]]
        Sorted wavelength columns, feature specs, and feature names.

    Raises
    ------
    ValueError
        If formatted feature names collide.
    """
    wavelength_columns_sorted = _sorted_wavelength_columns(columns)
    feature_specs = _build_feature_specs(wavelength_columns_sorted, frames)
    feature_names = [spec[-1] for spec in feature_specs]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate flattened feature names")
    return wavelength_columns_sorted, feature_specs, feature_names


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
    """
    if len(inputs) == 0:
        raise ValueError("inputs must contain at least one validated frame")

    if any(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
        if not all(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
            raise ValueError("flatten inputs must use one frame type")
        return _flatten_lazy_inputs(inputs)  # type: ignore[arg-type]

    sorted_inputs = sorted(inputs, key=lambda item: str(item[0].resolve()))
    wavelength_columns_sorted, feature_specs, feature_names = _prepare_flatten_layout(
        sorted_inputs[0][1].columns,
        [frame for _, frame in sorted_inputs],
    )

    rows: list[dict[str, object]] = []
    for path, frame in sorted_inputs:
        by_key = {
            (row["Step"], row["Sequence"], row["StepTime"]): row
            for row in frame.select(
                "Step", "Sequence", "StepTime", *wavelength_columns_sorted
            ).iter_rows(named=True)
        }
        row: dict[str, object] = {SOURCE_COLUMN: path.as_posix()}
        for column, step, sequence, step_time, name in feature_specs:
            match = by_key.get((step, sequence, step_time))
            row[name] = match[column] if match is not None else None
        rows.append(row)

    return pl.DataFrame(rows).select(SOURCE_COLUMN, *feature_names)


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
    _, feature_specs, feature_names = _prepare_flatten_layout(
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
                    for column, step, sequence, step_time, name in feature_specs
                ],
            )
        )
    return pl.concat(rows).select(SOURCE_COLUMN, *feature_names)
