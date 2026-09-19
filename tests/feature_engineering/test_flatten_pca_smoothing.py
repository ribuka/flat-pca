"""Tests for Flatten-PCA smoothing stages."""

from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering.preprocess.smoothing import (
    apply_t_smoothing as _apply_t_smoothing,
)
from flat_pca.feature_engineering.preprocess.smoothing import (
    apply_w_smoothing as _apply_w_smoothing,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def test_t_smoothing_uses_closed_real_time_windows_on_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Average adjacent real observations without assuming equal Time spacing."""
    frame = pl.read_parquet(real_fixture_paths[0]).sort("Time")
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    times = frame["Time"].to_list()
    gaps = [right - left for left, right in pairwise(times)]
    window = min(gaps)

    without_neighbors = _apply_t_smoothing(frame, window / 2)
    with_neighbors = _apply_t_smoothing(frame, window)
    expected = [
        frame.filter(
            (pl.col("Time") >= time - window) & (pl.col("Time") <= time + window)
        )[wavelength].mean()
        for time in times
    ]

    assert len(set(gaps)) > 1
    assert without_neighbors.equals(frame)
    assert with_neighbors[wavelength].to_list() == pytest.approx(expected)
    assert with_neighbors.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert with_neighbors.shape == frame.shape


def test_t_smoothing_keeps_step_and_sequence_groups_separate(
    real_fixture_paths: list[Path],
) -> None:
    """Keep nearby rows in different Step or Sequence groups isolated."""
    frame = (
        pl.read_parquet(real_fixture_paths[0])
        .head(4)
        .with_columns(
            pl.Series("Time", [0.0, 1.0, 1.0, 2.0]),
            pl.Series("Step", [0, 0, 1, 1]),
            pl.Series("Sequence", [0, 1, 0, 0]),
        )
    )
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )

    smoothed = _apply_t_smoothing(frame, 1.0)

    assert smoothed[wavelength][0] == pytest.approx(frame[wavelength][0])
    assert smoothed[wavelength][1] == pytest.approx(frame[wavelength][1])
    assert smoothed[wavelength][2] == pytest.approx(
        frame[wavelength].slice(2, 2).mean()
    )
    assert smoothed[wavelength][3] == pytest.approx(
        frame[wavelength].slice(2, 2).mean()
    )


def test_t_smoothing_none_preserves_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Return the validated real fixture unchanged when smoothing is disabled."""
    frame = pl.read_parquet(real_fixture_paths[0])

    assert _apply_t_smoothing(frame, None).equals(frame)


@pytest.mark.parametrize(
    "window",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_t_smoothing_rejects_invalid_windows(
    window: float, real_fixture_paths: list[Path]
) -> None:
    """Reject nonpositive and nonfinite t-smoothing half-window widths."""
    frame = pl.read_parquet(real_fixture_paths[0])

    with pytest.raises(ValueError, match="t_smoothing_window"):
        _apply_t_smoothing(frame, window)


def test_w_smoothing_uses_closed_real_wavelength_windows_on_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Average real wavelengths without assuming equal wavelength spacing."""
    frame = pl.read_parquet(real_fixture_paths[0]).head(3)
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    gaps = [right - left for left, right in pairwise(wavelengths)]
    window = min(gaps)

    smoothed = _apply_w_smoothing(frame, window)
    expected = [
        [
            frame.select(
                column
                for column, candidate in zip(wavelength_columns, wavelengths)
                if wavelength - window <= candidate <= wavelength + window
            ).row(row_index)
            for wavelength in wavelengths
        ]
        for row_index in range(frame.height)
    ]

    assert len(set(gaps)) > 1
    for row_index in range(frame.height):
        expected_means = [sum(values) / len(values) for values in expected[row_index]]
        assert smoothed.select(wavelength_columns).row(row_index) == pytest.approx(
            expected_means
        )
    assert smoothed.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert smoothed.shape == frame.shape


def test_w_smoothing_none_preserves_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Return the validated real fixture unchanged when smoothing is disabled."""
    frame = pl.read_parquet(real_fixture_paths[0])

    assert _apply_w_smoothing(frame, None).equals(frame)


@pytest.mark.parametrize(
    "window",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_w_smoothing_rejects_invalid_windows(
    window: float, real_fixture_paths: list[Path]
) -> None:
    """Reject nonpositive and nonfinite w-smoothing half-window widths."""
    frame = pl.read_parquet(real_fixture_paths[0])

    with pytest.raises(ValueError, match="w_smoothing_window"):
        _apply_w_smoothing(frame, window)
