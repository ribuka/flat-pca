"""Deterministic spectral flattening for Flatten-PCA."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .schema import parse_wavelength, wavelength_columns


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

    if any(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
        if not all(isinstance(frame, pl.LazyFrame) for _, frame in inputs):
            raise ValueError("flatten inputs must use one frame type")
        return _flatten_lazy_inputs(inputs)  # type: ignore[arg-type]

    rows: list[dict[str, object]] = []
    expected_feature_names: list[str] | None = None
    for path, frame in sorted(inputs, key=lambda item: str(item[0].resolve())):
        spectra = sorted(
            (
                (parse_wavelength(column), column)
                for column in wavelength_columns(frame.columns)
            ),
            key=lambda item: item[0],
        )
        sorted_frame = frame.sort("Step", "Sequence", "Time")
        feature_names: list[str] = []
        feature_values: list[object] = []
        for _, wavelength_column in spectra:
            for step, sequence, time, intensity in sorted_frame.select(
                "Step",
                "Sequence",
                "Time",
                wavelength_column,
            ).iter_rows():
                feature_names.append(
                    f"{wavelength_column}_{int(step)}_{int(sequence)}_{float(time):.2f}"
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
        If formatted feature names collide or input feature layouts differ.
    """
    first_frame = inputs[0][1]
    columns = first_frame.collect_schema().names()
    spectra = sorted(
        ((parse_wavelength(column), column) for column in wavelength_columns(columns)),
        key=lambda item: item[0],
    )
    metadata_rows = list(
        first_frame.select("Step", "Sequence", "Time")
        .unique()
        .sort("Step", "Sequence", "Time")
        .collect()
        .iter_rows()
    )
    feature_specs = [
        (
            column,
            step,
            sequence,
            time,
            f"{column}_{int(step)}_{int(sequence)}_{float(time):.2f}",
        )
        for _, column in spectra
        for step, sequence, time in metadata_rows
    ]
    feature_names = [spec[-1] for spec in feature_specs]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate flattened feature names")

    rows = []
    for path, frame in sorted(inputs, key=lambda item: str(item[0].resolve())):
        rows.append(
            frame.select(
                pl.lit(path.stem).alias("filename"),
                *[
                    pl.col(column)
                    .filter(
                        (pl.col("Step") == step)
                        & (pl.col("Sequence") == sequence)
                        & (pl.col("Time") == time)
                    )
                    .first()
                    .alias(name)
                    for column, step, sequence, time, name in feature_specs
                ],
            )
        )
    return pl.concat(rows).select("filename", *feature_names)
