"""Tests for deterministic Flatten-PCA feature flattening."""

from pathlib import Path

import polars as pl
import pytest

from spca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from spca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)
from spca.feature_engineering.flatten_pca.step_time import (
    add_step_time_columns as _add_step_time_columns,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence", "StepTime", "ReverseStepTime"}


def test_flatten_real_fixture_has_deterministic_columns_and_values(
    real_fixture_paths: list[Path],
) -> None:
    """Flatten one real Parquet fixture in numeric feature-key order."""
    path = real_fixture_paths[0]
    loaded = _load_and_validate_inputs([path])
    lazy_frame = _add_step_time_columns(loaded[0][1])
    frame = lazy_frame.collect()
    wavelength_columns = sorted(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )
    metadata_rows = sorted(
        frame.select("Step", "Sequence", "StepTime").iter_rows(),
        key=lambda row: tuple(float(value) for value in row),
    )
    expected_columns = ["filename"] + [
        f"{wavelength}_{int(step)}_{int(sequence)}_{float(step_time):.2f}"
        for wavelength in wavelength_columns
        for step, sequence, step_time in metadata_rows
    ]
    expected_values = [
        frame.filter(
            (pl.col("Step") == step)
            & (pl.col("Sequence") == sequence)
            & (pl.col("StepTime") == step_time)
        )[wavelength].item()
        for wavelength in wavelength_columns
        for step, sequence, step_time in metadata_rows
    ]

    flattened = _flatten_inputs([(path, lazy_frame)])
    assert isinstance(flattened, pl.LazyFrame)
    collected = flattened.collect()

    assert collected.height == 1
    assert collected.columns == expected_columns
    assert collected["filename"].to_list() == [path.stem]
    assert collected.row(0)[1:] == pytest.approx(expected_values)


def test_flatten_multiple_real_fixtures_ignores_input_order(
    real_fixture_paths: list[Path],
) -> None:
    """Combine real fixtures in deterministic normalized-path order."""
    paths = real_fixture_paths[:3]
    loaded = [
        (path, _add_step_time_columns(frame))
        for path, frame in _load_and_validate_inputs(paths)
    ]

    forward = _flatten_inputs(loaded)
    reverse = _flatten_inputs(list(reversed(loaded)))

    assert isinstance(forward, pl.LazyFrame)
    assert isinstance(reverse, pl.LazyFrame)
    collected = forward.collect()
    assert collected.equals(reverse.collect())
    assert collected["filename"].to_list() == [
        path.stem for path in sorted(path.resolve() for path in paths)
    ]


def test_flatten_sorts_step_and_sequence_and_rejects_name_collisions(
    real_fixture_paths: list[Path],
) -> None:
    """Sort numeric metadata keys and reject colliding formatted feature names."""
    path, lazy_fixture = _load_and_validate_inputs([real_fixture_paths[0]])[0]
    fixture = lazy_fixture.collect()
    frame = _add_step_time_columns(
        fixture.head(4).with_columns(
            pl.Series("Time", [1.0, 0.0, 1.0, 0.0]),
            pl.Series("Step", [1, 1, 0, 0]),
            pl.Series("Sequence", [1, 0, 1, 0]),
        )
    )

    flattened = _flatten_inputs([(path, frame)])
    first_wavelength = min(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )

    # Each row is its own singleton (Step, Sequence) segment, so StepTime is
    # 0.00 for every row regardless of its original Time value.
    assert flattened.columns[1:5] == [
        f"{first_wavelength}_0_0_0.00",
        f"{first_wavelength}_0_1_0.00",
        f"{first_wavelength}_1_0_0.00",
        f"{first_wavelength}_1_1_0.00",
    ]

    collision = _add_step_time_columns(
        fixture.head(2).with_columns(
            pl.Series("Time", [0.001, 0.002]),
            pl.lit(0).alias("Step"),
            pl.lit(0).alias("Sequence"),
        )
    )
    with pytest.raises(ValueError, match="duplicate flattened feature names"):
        _flatten_inputs([(path, collision)])
