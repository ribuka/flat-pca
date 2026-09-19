"""Tests for Step-row filtering, StepTime/ReverseStepTime, and edge trimming."""

from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering import preprocess_and_flatten
from flat_pca.feature_engineering.flatten_pca.input import load_and_validate_inputs
from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns,
    apply_edge_trim,
)

_META_COLUMNS = {"Time", "Step", "Sequence", "StepTime", "ReverseStepTime"}

# Every real fixture row has a constant Step of 1, so partial Step filtering
# and multi-segment StepTime boundaries cannot be exercised from the real
# fixture alone. These patterns overwrite the in-memory Step column derived
# from a real fixture and are written to `tmp_path` as required by the
# project's fixture policy.
_STEP_PATTERN_SPLIT = [1, 1, 1, 1, 2, 2, 2, 2]
_STEP_PATTERN_REVISIT = [1, 1, 2, 2, 1, 1, 2, 2]


def _write_stepped_fixture(
    source_path: Path,
    destination_path: Path,
    step_pattern: list[int],
) -> Path:
    """Write a real-fixture-derived Parquet with a replaced Step column.

    Parameters
    ----------
    source_path : Path
        Real fixture Parquet path to derive values from.
    destination_path : Path
        Destination path (under ``tmp_path``) for the derived file.
    step_pattern : list[int]
        Replacement ``Step`` values, one per row, in existing row order.

    Returns
    -------
    Path
        The written destination path.
    """
    frame = pl.read_parquet(source_path).with_columns(
        pl.Series("Step", step_pattern)
    )
    frame.write_parquet(destination_path)
    return destination_path


def test_target_steps_filters_rows_and_reduces_flatten_columns(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Keep only the requested Step values and shrink the feature-column set."""
    paths = [
        _write_stepped_fixture(
            path, tmp_path / path.name, _STEP_PATTERN_SPLIT
        )
        for path in real_fixture_paths[:2]
    ]

    full = preprocess_and_flatten(paths).collect()
    filtered = preprocess_and_flatten(paths, target_steps=[1]).collect()

    assert filtered.width == (full.width - 1) // 2 + 1
    for column in filtered.columns[1:]:
        assert column.split("_")[1] == "1"


def test_target_steps_none_preserves_existing_result(
    real_fixture_paths: list[Path],
) -> None:
    """Match the unfiltered flatten result when ``target_steps`` is ``None``."""
    default = preprocess_and_flatten(real_fixture_paths[:2]).collect()
    explicit_none = preprocess_and_flatten(
        real_fixture_paths[:2], target_steps=None
    ).collect()

    assert default.equals(explicit_none)


def test_add_step_time_columns_resets_per_segment_and_reverses_correctly(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Restart StepTime at each Step/Sequence boundary, including a revisit."""
    path = _write_stepped_fixture(
        real_fixture_paths[0], tmp_path / real_fixture_paths[0].name,
        _STEP_PATTERN_REVISIT,
    )
    _, lazy_frame = load_and_validate_inputs([path])[0]
    frame = add_step_time_columns(lazy_frame).collect()

    assert frame.columns[:4] == ["Time", "StepTime", "ReverseStepTime", "Step"]

    time = frame.get_column("Time").to_list()
    step_time = frame.get_column("StepTime").to_list()
    reverse_step_time = frame.get_column("ReverseStepTime").to_list()

    expected_step_time: list[float] = []
    expected_reverse_step_time: list[float] = []
    for segment_start in range(0, 8, 2):
        segment_times = time[segment_start : segment_start + 2]
        base, end = segment_times[0], segment_times[-1]
        expected_step_time.extend(t - base for t in segment_times)
        expected_reverse_step_time.extend(end - t for t in segment_times)

    assert step_time == pytest.approx(expected_step_time)
    assert reverse_step_time == pytest.approx(expected_reverse_step_time)
    # The first row of every segment starts at StepTime 0.00.
    assert all(step_time[index] == 0.0 for index in (0, 2, 4, 6))
    # The last row of every segment always ends at ReverseStepTime 0.00.
    assert all(reverse_step_time[index] == 0.0 for index in (1, 3, 5, 7))


def test_flatten_feature_names_use_step_time_not_time(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Derive the flattened feature-name ``t`` part from StepTime, not Time."""
    path = _write_stepped_fixture(
        real_fixture_paths[0], tmp_path / real_fixture_paths[0].name,
        _STEP_PATTERN_REVISIT,
    )

    flattened = preprocess_and_flatten([path]).collect()

    # Row index 2 has Time=19380.0 but starts a new Step=2 segment, so its
    # StepTime, not its Time, is 0.00.
    assert any(column.endswith("_2_1_0.00") for column in flattened.columns[1:])
    assert not any("19380.00" in column for column in flattened.columns[1:])


def test_apply_edge_trim_nulls_non_meta_columns_near_segment_edges(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Null spectral values within either trimmed edge and keep meta columns."""
    path = _write_stepped_fixture(
        real_fixture_paths[0], tmp_path / real_fixture_paths[0].name,
        _STEP_PATTERN_SPLIT,
    )
    _, lazy_frame = load_and_validate_inputs([path])[0]
    frame = add_step_time_columns(lazy_frame).collect()
    wavelength_columns = [
        column for column in frame.columns if column not in _META_COLUMNS
    ]

    trimmed = apply_edge_trim(frame, [5000.0, 5000.0])
    assert isinstance(trimmed, pl.DataFrame)

    step_time = trimmed.get_column("StepTime").to_list()
    reverse_step_time = trimmed.get_column("ReverseStepTime").to_list()
    sample_values = trimmed.get_column(wavelength_columns[0]).to_list()

    edge_trimmed_seen = False
    preserved_seen = False
    for row_step_time, row_reverse_step_time, value in zip(
        step_time, reverse_step_time, sample_values, strict=True
    ):
        if row_step_time < 5000.0 or row_reverse_step_time < 5000.0:
            assert value is None
            edge_trimmed_seen = True
        else:
            assert value is not None
            preserved_seen = True
    assert edge_trimmed_seen
    assert preserved_seen

    for column in _META_COLUMNS:
        assert trimmed.get_column(column).null_count() == 0

    unchanged = apply_edge_trim(frame, None)
    assert unchanged.equals(frame)


def test_preprocess_and_flatten_end_to_end_with_target_steps_and_edge_trim(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Combine target_steps, edge_trim, and downsampling on derived real data."""
    paths = [
        _write_stepped_fixture(path, tmp_path / path.name, _STEP_PATTERN_SPLIT)
        for path in real_fixture_paths
    ]
    original_values = {
        path.stem: pl.read_parquet(path).get_column("649.9nm").to_list()
        for path in real_fixture_paths
    }

    flattened = preprocess_and_flatten(
        paths,
        target_steps=[1],
        edge_trim=[5000.0, 5000.0],
        t_downsampling_stride=2,
        w_downsampling_stride=2,
    ).collect()

    assert flattened.width == 1 + 2 * 8
    assert "649.9nm_1_1_0.00" in flattened.columns
    assert "649.9nm_1_1_19380.00" in flattened.columns

    for row in flattened.iter_rows(named=True):
        assert row["649.9nm_1_1_0.00"] is None
        assert row["649.9nm_1_1_19380.00"] == pytest.approx(
            original_values[row["filename"]][2]
        )
