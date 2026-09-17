"""Tests for Flatten-PCA downsampling stages."""

from pathlib import Path

import polars as pl
import pytest

from spca.feature_engineering.flatten_pca.downsampling import (
    apply_t_downsampling as _apply_t_downsampling,
)
from spca.feature_engineering.flatten_pca.downsampling import (
    collect_unique_times as _collect_unique_times,
)


def test_t_downsampling_collects_sorted_unique_times_from_all_real_inputs(
    real_fixture_paths: list[Path],
) -> None:
    """Collect one numeric ascending Unique array from every real fixture."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]

    unique_times = _collect_unique_times(frames)

    expected = sorted(
        {
            float(time)
            for frame in frames
            for time in frame.get_column("Time").to_list()
        }
    )
    assert unique_times == expected
    assert all(isinstance(time, float) for time in unique_times)


@pytest.mark.parametrize("stride", [1, 2])
def test_t_downsampling_keeps_selected_real_rows_for_every_group(
    stride: int,
    real_fixture_paths: list[Path],
) -> None:
    """Keep all real Step and Sequence rows at selected Time indices."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]
    unique_times = _collect_unique_times(frames)
    frame = frames[0]
    selected_times = unique_times[::stride]

    downsampled = _apply_t_downsampling(frame, unique_times, stride)
    expected = frame.filter(pl.col("Time").is_in(selected_times))

    assert downsampled.equals(expected)
    assert downsampled.get_column("Time").unique().sort().to_list() == selected_times
    assert downsampled.select("Time", "Step", "Sequence").equals(
        expected.select("Time", "Step", "Sequence")
    )


def test_t_downsampling_does_not_add_unselected_tail(
    real_fixture_paths: list[Path],
) -> None:
    """Drop a real-data-derived final Time when its index is not selected."""
    frame = pl.read_parquet(real_fixture_paths[0]).head(6)
    unique_times = _collect_unique_times([frame])

    downsampled = _apply_t_downsampling(frame, unique_times, 2)

    assert downsampled.get_column("Time").unique().sort().to_list() == unique_times[::2]
    assert unique_times[-1] not in downsampled.get_column("Time").to_list()


@pytest.mark.parametrize("stride", [0, -1, True, False, 1.5, "2", None])
def test_t_downsampling_rejects_invalid_stride(
    stride: object,
    real_fixture_paths: list[Path],
) -> None:
    """Reject zero, negative, boolean, and non-integer intervals."""
    frame = pl.read_parquet(real_fixture_paths[0])
    unique_times = _collect_unique_times([frame])

    with pytest.raises(ValueError, match="t_downsampling_stride"):
        _apply_t_downsampling(frame, unique_times, stride)  # type: ignore[arg-type]
