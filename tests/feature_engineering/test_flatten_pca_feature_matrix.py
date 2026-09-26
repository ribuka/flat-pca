"""Tests for the NumPy feature-matrix flattening and fused sparse pruning."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.flatten_pca.feature_matrix import (
    build_flattened_frame as _build_flattened_frame,
)
from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_and_prune_inputs as _flatten_and_prune_inputs,
)
from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns as _add_step_time_columns,
)
from flat_pca.feature_engineering.preprocess import (
    drop_sparse_feature_columns as _drop_sparse_feature_columns,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence", "StepTime", "ReverseStepTime"}


def _canonical_fixture_frame(path: Path) -> pl.DataFrame:
    """Read one real fixture and generate its StepTime columns.

    Parameters
    ----------
    path : Path
        Real spectral Parquet fixture path.

    Returns
    -------
    pl.DataFrame
        Fixture contents with ``StepTime`` and ``ReverseStepTime`` added.
    """
    canonical = _add_step_time_columns(pl.read_parquet(path))
    assert isinstance(canonical, pl.DataFrame)
    return canonical


def _build_shared_grid_inputs(
    real_fixture_paths: list[Path],
) -> list[tuple[Path, pl.DataFrame]]:
    """Pair real fixtures that all share one metadata grid with their paths.

    Parameters
    ----------
    real_fixture_paths : list[Path]
        Real fixture Parquet paths.

    Returns
    -------
    list[tuple[Path, pl.DataFrame]]
        Three inputs built from the same file's grid, so no feature column
        is missing for any input.
    """
    frame = _canonical_fixture_frame(real_fixture_paths[0])
    return [(path, frame) for path in real_fixture_paths[:3]]


def _build_mismatched_grid_inputs(
    real_fixture_paths: list[Path],
) -> list[tuple[Path, pl.DataFrame]]:
    """Build four inputs whose metadata grids differ in a controlled way.

    Every variant keeps the minimum ``Time`` row, so the per-segment
    ``StepTime`` origin never shifts between variants. The last row's
    combination is missing from 2 of 4 variants (a 50% null ratio) and the
    second-to-last row's combination from 1 of 4 (25%).

    Parameters
    ----------
    real_fixture_paths : list[Path]
        Real fixture Parquet paths; only the first file's data is used,
        paired with four distinct path labels.

    Returns
    -------
    list[tuple[Path, pl.DataFrame]]
        Four flatten inputs with differing metadata grids.
    """
    frame = pl.read_parquet(real_fixture_paths[0])
    last_time = frame["Time"][frame.height - 1]
    second_last_time = frame["Time"][frame.height - 2]
    variants = [
        frame,
        frame.filter(pl.col("Time") != last_time),
        frame.filter(pl.col("Time") != last_time),
        frame.filter(pl.col("Time") != second_last_time),
    ]
    inputs: list[tuple[Path, pl.DataFrame]] = []
    for path, variant in zip(real_fixture_paths[:4], variants, strict=True):
        with_step_time = _add_step_time_columns(variant)
        assert isinstance(with_step_time, pl.DataFrame)
        inputs.append((path, with_step_time))
    return inputs


def test_eager_flatten_matches_the_deferred_flatten_on_mismatched_grids(
    real_fixture_paths: list[Path],
) -> None:
    """Match the unchanged deferred flatten column for column and value for value."""
    inputs = _build_mismatched_grid_inputs(real_fixture_paths)

    eager = _flatten_inputs(inputs)
    deferred = _flatten_inputs([(path, frame.lazy()) for path, frame in inputs])

    assert isinstance(eager, pl.DataFrame)
    assert isinstance(deferred, pl.LazyFrame)
    assert eager.equals(deferred.collect())


@pytest.mark.parametrize(
    ("max_null_ratio", "expected_combination_count"),
    [(0.0, 6), (0.1, 6), (0.25, 7), (1.0, 8)],
)
def test_flatten_and_prune_matches_flatten_then_prune(
    real_fixture_paths: list[Path],
    max_null_ratio: float,
    expected_combination_count: int,
) -> None:
    """Fuse flatten and pruning without changing the pruned result."""
    inputs = _build_mismatched_grid_inputs(real_fixture_paths)
    flattened = _flatten_inputs(inputs)
    assert isinstance(flattened, pl.DataFrame)
    expected = _drop_sparse_feature_columns(flattened, max_null_ratio)
    assert isinstance(expected, pl.DataFrame)
    wavelength_count = sum(
        1 for column in inputs[0][1].columns if column not in METADATA_COLUMNS
    )

    fused = _flatten_and_prune_inputs(inputs, max_null_ratio)

    assert fused.equals(expected)
    assert fused.width == 1 + wavelength_count * expected_combination_count


def test_flatten_and_prune_matches_flatten_then_prune_on_a_shared_grid(
    real_fixture_paths: list[Path],
) -> None:
    """Keep every feature column when no input is missing a combination."""
    inputs = _build_shared_grid_inputs(real_fixture_paths)
    flattened = _flatten_inputs(inputs)
    assert isinstance(flattened, pl.DataFrame)

    fused = _flatten_and_prune_inputs(inputs, 0.0)

    assert fused.equals(flattened)


def test_flatten_and_prune_represents_missing_combinations_as_null(
    real_fixture_paths: list[Path],
) -> None:
    """Restore missing combinations as polars nulls rather than NaN."""
    inputs = _build_mismatched_grid_inputs(real_fixture_paths)

    fused = _flatten_and_prune_inputs(inputs, 1.0)

    features = fused.drop("source")
    assert features.null_count().sum_horizontal().item() > 0
    assert all(dtype == pl.Float64 for dtype in features.dtypes)
    assert not any(features.select(pl.all().is_nan().any()).row(0))


def test_flatten_and_prune_keeps_the_last_duplicate_metadata_row(
    real_fixture_paths: list[Path],
) -> None:
    """Let the last row win when one frame repeats a metadata combination."""
    path = real_fixture_paths[0]
    wavelength = next(
        column
        for column in pl.read_parquet(path).columns
        if column not in METADATA_COLUMNS
    )
    duplicated = pl.DataFrame(
        {
            "Time": [0.0, 0.0],
            "StepTime": [0.0, 0.0],
            "ReverseStepTime": [0.0, 0.0],
            "Step": [1, 1],
            "Sequence": [1, 1],
            wavelength: [10.0, 20.0],
        }
    )

    fused = _flatten_and_prune_inputs([(path, duplicated)], 0.0)

    assert fused.width == 2
    assert fused.row(0)[1] == 20.0


def _single_wavelength_frame(
    wavelength_values: pl.Series,
) -> pl.DataFrame:
    """Build a two-combination frame carrying one wavelength column.

    Parameters
    ----------
    wavelength_values : pl.Series
        Two spectral values named after a canonical wavelength column.

    Returns
    -------
    pl.DataFrame
        Frame with the metadata columns flattening requires plus
        ``wavelength_values``.
    """
    return pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "StepTime": [0.0, 1.0],
            "ReverseStepTime": [1.0, 0.0],
            "Step": [1, 1],
            "Sequence": [1, 1],
        }
    ).with_columns(wavelength_values)


def test_flatten_keeps_integer_spectral_dtype_and_exact_values(
    real_fixture_paths: list[Path],
) -> None:
    """Route integer wavelength columns around the lossy float64 matrix.

    A ``float64`` feature matrix silently rounds integers wider than 53
    bits, so integer inputs must keep the row-dict flattening that preserves
    both the ``Int64`` dtype and the exact value.
    """
    beyond_float64_precision = 2**53 + 1
    frame = _single_wavelength_frame(
        pl.Series("649.9nm", [beyond_float64_precision, 1], dtype=pl.Int64)
    )
    inputs = [(real_fixture_paths[0], frame)]

    flattened = _flatten_inputs(inputs)
    fused = _flatten_and_prune_inputs(inputs, 0.0)

    assert isinstance(flattened, pl.DataFrame)
    assert flattened.equals(fused)
    assert fused.dtypes[1:] == [pl.Int64, pl.Int64]
    assert fused.row(0)[1] == beyond_float64_precision


def test_flatten_widens_float32_spectral_columns_to_float64(
    real_fixture_paths: list[Path],
) -> None:
    """Widen Float32 inputs exactly as the row-dict flattening always did."""
    frame = _single_wavelength_frame(
        pl.Series("649.9nm", [1.5, 2.5], dtype=pl.Float32)
    )
    inputs = [(real_fixture_paths[0], frame)]

    fused = _flatten_and_prune_inputs(inputs, 0.0)

    assert fused.dtypes[1:] == [pl.Float64, pl.Float64]
    assert fused.row(0)[1:] == (1.5, 2.5)


def test_flatten_and_prune_rejects_invalid_arguments(
    real_fixture_paths: list[Path],
) -> None:
    """Reject empty inputs, deferred frames, and out-of-range thresholds."""
    inputs = _build_shared_grid_inputs(real_fixture_paths)

    with pytest.raises(ValueError, match="at least one validated frame"):
        _flatten_and_prune_inputs([], 0.1)
    with pytest.raises(ValueError, match="must be materialized frames"):
        _flatten_and_prune_inputs(
            [(path, frame.lazy()) for path, frame in inputs],  # ty:ignore[invalid-argument-type]
            0.1,
        )
    with pytest.raises(ValueError, match="threshold must be between"):
        _flatten_and_prune_inputs(inputs, 1.5)


def _row_oriented_reference(
    matrix: np.ndarray, sources: list[str], feature_names: list[str]
) -> pl.DataFrame:
    """Build the flattened frame with the direct row-oriented construction.

    Parameters
    ----------
    matrix : np.ndarray
        Feature matrix with ``NaN`` for missing values.
    sources : list[str]
        Source texts, one per matrix row.
    feature_names : list[str]
        Feature names, one per matrix column.

    Returns
    -------
    pl.DataFrame
        Reference frame that ``build_flattened_frame`` must equal.
    """
    frame = pl.DataFrame(matrix, schema=feature_names, orient="row", nan_to_null=True)
    return frame.insert_column(0, pl.Series("source", sources, dtype=pl.String))


@pytest.mark.parametrize(
    ("row_count", "column_count"),
    [(3, 5), (1, 4), (3, 0), (0, 4), (0, 0)],
)
def test_build_flattened_frame_matches_the_row_oriented_construction(
    row_count: int, column_count: int
) -> None:
    """Transpose-based widening must equal the row-oriented construction."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(row_count, column_count))
    matrix[rng.random(matrix.shape) < 0.3] = np.nan
    if matrix.size:
        matrix[:, 0] = np.nan
    sources = [f"input_{index}.parquet" for index in range(row_count)]
    feature_names = [f"feature_{index}" for index in range(column_count)]

    result = _build_flattened_frame(matrix, sources, feature_names)

    expected = _row_oriented_reference(matrix, sources, feature_names)
    assert result.equals(expected)
    assert result.schema == expected.schema
    assert all(result[name].is_nan().sum() == 0 for name in feature_names)
