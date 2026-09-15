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

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def test_flatten_real_fixture_has_deterministic_columns_and_values(
    real_fixture_paths: list[Path],
) -> None:
    """Flatten one real Parquet fixture in numeric feature-key order."""
    path = real_fixture_paths[0]
    loaded = _load_and_validate_inputs([path])
    frame = loaded[0][1]
    wavelength_columns = sorted(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )
    metadata_rows = sorted(
        frame.select("Step", "Sequence", "Time").iter_rows(),
        key=lambda row: tuple(float(value) for value in row),
    )
    expected_columns = ["filename"] + [
        f"{wavelength}_{int(step)}_{int(sequence)}_{float(time):.2f}"
        for wavelength in wavelength_columns
        for step, sequence, time in metadata_rows
    ]
    expected_values = [
        frame.filter(
            (pl.col("Step") == step)
            & (pl.col("Sequence") == sequence)
            & (pl.col("Time") == time)
        )[wavelength].item()
        for wavelength in wavelength_columns
        for step, sequence, time in metadata_rows
    ]

    flattened = _flatten_inputs(loaded)

    assert flattened.height == 1
    assert flattened.columns == expected_columns
    assert flattened["filename"].to_list() == [path.stem]
    assert flattened.row(0)[1:] == pytest.approx(expected_values)


def test_flatten_multiple_real_fixtures_ignores_input_order(
    real_fixture_paths: list[Path],
) -> None:
    """Combine real fixtures in deterministic normalized-path order."""
    paths = real_fixture_paths[:3]
    loaded = _load_and_validate_inputs(paths)

    forward = _flatten_inputs(loaded)
    reverse = _flatten_inputs(list(reversed(loaded)))

    assert forward.equals(reverse)
    assert forward["filename"].to_list() == [
        path.stem for path in sorted(path.resolve() for path in paths)
    ]


def test_flatten_sorts_step_and_sequence_and_rejects_name_collisions(
    real_fixture_paths: list[Path],
) -> None:
    """Sort numeric metadata keys and reject colliding formatted feature names."""
    path, fixture = _load_and_validate_inputs([real_fixture_paths[0]])[0]
    frame = fixture.head(4).with_columns(
        pl.Series("Time", [1.0, 0.0, 1.0, 0.0]),
        pl.Series("Step", [1, 1, 0, 0]),
        pl.Series("Sequence", [1, 0, 1, 0]),
    )

    flattened = _flatten_inputs([(path, frame)])
    first_wavelength = min(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )

    assert flattened.columns[1:5] == [
        f"{first_wavelength}_0_0_0.00",
        f"{first_wavelength}_0_1_1.00",
        f"{first_wavelength}_1_0_0.00",
        f"{first_wavelength}_1_1_1.00",
    ]

    collision = fixture.head(2).with_columns(
        pl.Series("Time", [0.001, 0.002]),
        pl.lit(0).alias("Step"),
        pl.lit(0).alias("Sequence"),
    )
    with pytest.raises(ValueError, match="duplicate flattened feature names"):
        _flatten_inputs([(path, collision)])
