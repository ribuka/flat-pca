"""Tests for deterministic Flatten-PCA feature flattening."""

from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_and_prune_inputs as _flatten_and_prune_inputs,
)
from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from flat_pca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)
from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns as _add_step_time_columns,
)
from flat_pca.feature_engineering.preprocess.sparse_columns import (
    drop_sparse_feature_columns as _drop_sparse_feature_columns,
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
    expected_columns = ["source"] + [
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
    assert collected["source"].to_list() == [path.as_posix()]
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
    assert collected["source"].to_list() == [
        path.as_posix() for path in sorted(path.resolve() for path in paths)
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


def _build_union_gap_fixtures(
    real_fixture_paths: list[Path],
) -> tuple[Path, pl.DataFrame, Path, pl.DataFrame, str]:
    """Derive two real-fixture frames where one is missing one metadata combo.

    Returns
    -------
    tuple[Path, pl.DataFrame, Path, pl.DataFrame, str]
        ``path_a``, the full frame, ``path_b``, a frame missing its last
        ``(Step, Sequence, StepTime)`` row, and the formatted feature name
        for the combination ``path_b`` lacks.
    """
    path_a, path_b = real_fixture_paths[0], real_fixture_paths[1]
    frame_a = _add_step_time_columns(_load_and_validate_inputs([path_a])[0][1]).collect()
    frame_b_full = _add_step_time_columns(_load_and_validate_inputs([path_b])[0][1]).collect()
    frame_b = frame_b_full.slice(0, frame_b_full.height - 1)
    removed = frame_b_full.select("Step", "Sequence", "StepTime").row(
        frame_b_full.height - 1, named=True
    )
    wavelength = next(
        column for column in frame_a.columns if column not in METADATA_COLUMNS
    )
    missing_column = (
        f"{wavelength}_{int(removed['Step'])}_{int(removed['Sequence'])}"
        f"_{float(removed['StepTime']):.2f}"
    )
    return path_a, frame_a, path_b, frame_b, missing_column


def test_flatten_lazy_inputs_unions_metadata_grids_and_fills_null_gaps(
    real_fixture_paths: list[Path],
) -> None:
    """Union differing per-file metadata grids and null-fill missing combos."""
    path_a, frame_a, path_b, frame_b, missing_column = _build_union_gap_fixtures(
        real_fixture_paths
    )

    flattened = _flatten_inputs(
        [(path_a, frame_a.lazy()), (path_b, frame_b.lazy())]
    )
    assert isinstance(flattened, pl.LazyFrame)
    collected = flattened.collect()
    row_by_source = {row["source"]: row for row in collected.iter_rows(named=True)}

    wavelength_column_count = sum(
        1 for column in frame_a.columns if column not in METADATA_COLUMNS
    )

    assert missing_column in collected.columns
    assert row_by_source[path_a.as_posix()][missing_column] is not None
    assert row_by_source[path_b.as_posix()][missing_column] is None
    assert all(
        value is not None
        for key, value in row_by_source[path_a.as_posix()].items()
        if key != "source"
    )
    # The removed row drops one (Step, Sequence, StepTime) combo entirely, so
    # every wavelength column at that combo becomes null for path_b's row.
    assert (
        sum(
            1
            for key, value in row_by_source[path_b.as_posix()].items()
            if key != "source" and value is None
        )
        == wavelength_column_count
    )


def test_flatten_eager_inputs_unions_metadata_grids_and_fills_null_gaps(
    real_fixture_paths: list[Path],
) -> None:
    """Union differing per-file metadata grids for eager DataFrame inputs too."""
    path_a, frame_a, path_b, frame_b, missing_column = _build_union_gap_fixtures(
        real_fixture_paths
    )

    flattened = _flatten_inputs([(path_a, frame_a), (path_b, frame_b)])
    assert isinstance(flattened, pl.DataFrame)
    row_by_source = {row["source"]: row for row in flattened.iter_rows(named=True)}

    assert missing_column in flattened.columns
    assert row_by_source[path_a.as_posix()][missing_column] is not None
    assert row_by_source[path_b.as_posix()][missing_column] is None


def _load_step_time_frames(paths: list[Path]) -> list[tuple[Path, pl.DataFrame]]:
    """Load real fixtures with StepTime columns as materialized frames."""
    return [
        (path, _add_step_time_columns(frame).collect())
        for path, frame in _load_and_validate_inputs(paths)
    ]


@pytest.mark.parametrize("grids", ["matched", "mismatched"])
@pytest.mark.parametrize("max_null_ratio", [0.0, 0.1, 0.25, 1.0])
def test_flatten_lazy_inputs_matches_materialized_flatten_and_prune(
    real_fixture_paths: list[Path],
    tmp_path: Path,
    grids: str,
    max_null_ratio: float,
) -> None:
    """Produce exactly the materialized branch's pruned output when deferred."""
    if grids == "matched":
        _, frame = _load_step_time_frames(real_fixture_paths[:1])[0]
        inputs = [(tmp_path / f"{index}.parquet", frame) for index in range(3)]
    else:
        inputs = _load_step_time_frames(real_fixture_paths)

    materialized = _flatten_and_prune_inputs(inputs, max_null_ratio)
    deferred = _flatten_inputs([(path, frame.lazy()) for path, frame in inputs])
    assert isinstance(deferred, pl.LazyFrame)
    pruned = _drop_sparse_feature_columns(deferred, max_null_ratio)
    assert isinstance(pruned, pl.LazyFrame)

    assert pruned.collect().equals(materialized)


def test_flatten_duplicate_metadata_rows_keep_last_row_in_both_branches(
    tmp_path: Path,
) -> None:
    """Keep the last row of a repeated combination whether deferred or not."""
    frame = pl.DataFrame(
        {
            "Step": [0, 0, 0],
            "Sequence": [0, 0, 0],
            "StepTime": [0.0, 0.0, 1.0],
            "500.0nm": [10.0, 20.0, 30.0],
        }
    )
    path = tmp_path / "duplicate.parquet"

    materialized = _flatten_inputs([(path, frame)])
    deferred = _flatten_inputs([(path, frame.lazy())])
    assert isinstance(materialized, pl.DataFrame)
    assert isinstance(deferred, pl.LazyFrame)
    collected = deferred.collect()

    assert collected.equals(materialized)
    assert collected["500.0nm_0_0_0.00"].to_list() == [20.0]


def test_flatten_lazy_inputs_widens_float32_and_nulls_nan(tmp_path: Path) -> None:
    """Widen Float32 to Float64 and turn NaN into null like the materialized branch."""
    frame = pl.DataFrame(
        {
            "Step": [0, 0],
            "Sequence": [0, 0],
            "StepTime": [0.0, 1.0],
            "500.0nm": pl.Series([1.5, float("nan")], dtype=pl.Float32),
        }
    )
    path = tmp_path / "float32.parquet"

    materialized = _flatten_inputs([(path, frame)])
    deferred = _flatten_inputs([(path, frame.lazy())])
    assert isinstance(deferred, pl.LazyFrame)
    collected = deferred.collect()

    assert collected.equals(materialized)
    assert collected.schema["500.0nm_0_0_0.00"] == pl.Float64
    assert collected["500.0nm_0_0_1.00"].to_list() == [None]


def test_flatten_rejects_mixed_frame_types(tmp_path: Path) -> None:
    """Reject inputs that mix DataFrame and LazyFrame frames."""
    frame = pl.DataFrame(
        {"Step": [0], "Sequence": [0], "StepTime": [0.0], "500.0nm": [1.0]}
    )

    with pytest.raises(ValueError, match="flatten inputs must use one frame type"):
        _flatten_inputs(
            [(tmp_path / "a.parquet", frame), (tmp_path / "b.parquet", frame.lazy())]
        )


def test_flatten_lazy_inputs_keeps_integer_precision_beside_float_columns(
    tmp_path: Path,
) -> None:
    """Keep an Int64 feature exact when another wavelength column is Float64."""
    exact_value = 2**53 + 1  # not exactly representable as a float64
    frame = pl.DataFrame(
        {
            "Step": [0],
            "Sequence": [0],
            "StepTime": [0.0],
            "500.0nm": pl.Series([exact_value], dtype=pl.Int64),
            "600.0nm": pl.Series([1.5], dtype=pl.Float64),
            "700.0nm": pl.Series([7], dtype=pl.Int32),
        }
    )
    path = tmp_path / "mixed.parquet"

    materialized = _flatten_inputs([(path, frame)])
    deferred = _flatten_inputs([(path, frame.lazy())])
    assert isinstance(deferred, pl.LazyFrame)
    collected = deferred.collect()

    assert collected.equals(materialized)
    assert collected.schema == deferred.collect_schema()
    assert collected.schema["500.0nm_0_0_0.00"] == pl.Int64
    assert collected["500.0nm_0_0_0.00"].to_list() == [exact_value]


def test_flatten_lazy_inputs_uses_supertype_of_wavelength_dtypes_across_inputs(
    tmp_path: Path,
) -> None:
    """Keep a later input's Float64 value when the first input stores Int64."""
    integer_frame = pl.DataFrame(
        {
            "Step": [0],
            "Sequence": [0],
            "StepTime": [0.0],
            "500.0nm": pl.Series([1], dtype=pl.Int64),
        }
    )
    float_frame = integer_frame.with_columns(
        pl.Series("500.0nm", [1.5], dtype=pl.Float64)
    )
    inputs = [(tmp_path / "a.parquet", integer_frame), (tmp_path / "b.parquet", float_frame)]

    for ordered in (inputs, list(reversed(inputs))):
        materialized = _flatten_inputs(ordered)
        deferred = _flatten_inputs([(path, frame.lazy()) for path, frame in ordered])
        assert isinstance(deferred, pl.LazyFrame)
        collected = deferred.collect()

        assert collected.equals(materialized)
        assert collected.schema["500.0nm_0_0_0.00"] == pl.Float64
        assert collected["500.0nm_0_0_0.00"].to_list() == [1.0, 1.5]


def test_flatten_lazy_inputs_keeps_metadata_keys_distinct_across_dtypes(
    tmp_path: Path,
) -> None:
    """Keep StepTime keys distinct that would collide if cast to Float32."""
    narrow_frame = pl.DataFrame(
        {
            "Step": [0],
            "Sequence": [0],
            "StepTime": pl.Series([16_777_216.0], dtype=pl.Float32),
            "500.0nm": [1.0],
        }
    )
    wide_frame = pl.DataFrame(
        {
            "Step": [0],
            "Sequence": [0],
            "StepTime": pl.Series([16_777_217.0], dtype=pl.Float64),
            "500.0nm": [2.0],
        }
    )
    inputs = [(tmp_path / "a.parquet", narrow_frame), (tmp_path / "b.parquet", wide_frame)]

    materialized = _flatten_inputs(inputs)
    deferred = _flatten_inputs([(path, frame.lazy()) for path, frame in inputs])
    assert isinstance(deferred, pl.LazyFrame)
    collected = deferred.collect()

    assert collected.equals(materialized)
    assert collected["500.0nm_0_0_16777216.00"].to_list() == [1.0, None]
    assert collected["500.0nm_0_0_16777217.00"].to_list() == [None, 2.0]


def test_flatten_lazy_inputs_keeps_signed_and_unsigned_step_keys_distinct(
    tmp_path: Path,
) -> None:
    """Keep Int64 and UInt64 Step keys distinct that would collide in Float64."""
    signed_frame = pl.DataFrame(
        {
            "Step": pl.Series([2**63 - 1], dtype=pl.Int64),
            "Sequence": [0],
            "StepTime": [0.0],
            "500.0nm": [1.0],
        }
    )
    unsigned_frame = pl.DataFrame(
        {
            "Step": pl.Series([2**63], dtype=pl.UInt64),
            "Sequence": [0],
            "StepTime": [0.0],
            "500.0nm": [2.0],
        }
    )
    inputs = [
        (tmp_path / "a.parquet", signed_frame),
        (tmp_path / "b.parquet", unsigned_frame),
    ]

    materialized = _flatten_inputs(inputs)
    deferred = _flatten_inputs([(path, frame.lazy()) for path, frame in inputs])
    assert isinstance(deferred, pl.LazyFrame)
    collected = deferred.collect()

    assert collected.equals(materialized)
    assert collected[f"500.0nm_{2**63 - 1}_0_0.00"].to_list() == [1.0, None]
    assert collected[f"500.0nm_{2**63}_0_0.00"].to_list() == [None, 2.0]
