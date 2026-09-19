"""Tests for Flatten-PCA normalization stages."""

from pathlib import Path

import polars as pl
import pytest

from spca.feature_engineering.preprocess.normalization import (
    apply_t_normalization as _apply_t_normalization,
)
from spca.feature_engineering.preprocess.normalization import (
    apply_w_normalization as _apply_w_normalization,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def test_t_normalization_makes_real_fixture_reference_mean_one(
    real_fixture_paths: list[Path],
) -> None:
    """Normalize every spectrum by its inclusive real-Time reference mean."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    times = sorted(frame["Time"].unique().to_list())
    reference_range = (times[1], times[-2])

    normalized = _apply_t_normalization(frame, reference_range)
    reference = normalized.filter(
        pl.col("Time").is_between(*reference_range, closed="both")
    )

    assert reference.select(wavelength_columns).mean().row(0) == pytest.approx(
        [1.0] * len(wavelength_columns)
    )
    assert normalized.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert normalized.shape == frame.shape


def test_t_normalization_keeps_step_and_sequence_groups_separate(
    real_fixture_paths: list[Path],
) -> None:
    """Use a distinct reference mean for each Step and Sequence group."""
    frame = (
        pl.read_parquet(real_fixture_paths[0])
        .head(4)
        .with_columns(
            pl.Series("Time", [0.0, 1.0, 0.0, 1.0]),
            pl.Series("Step", [0, 0, 1, 1]),
            pl.Series("Sequence", [0, 0, 1, 1]),
        )
    )
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )

    normalized = _apply_t_normalization(frame, (0.0, 0.0))

    assert normalized[wavelength][0] == pytest.approx(1.0)
    assert normalized[wavelength][1] == pytest.approx(
        frame[wavelength][1] / frame[wavelength][0]
    )
    assert normalized[wavelength][2] == pytest.approx(1.0)
    assert normalized[wavelength][3] == pytest.approx(
        frame[wavelength][3] / frame[wavelength][2]
    )


def test_t_normalization_none_preserves_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Return the validated real fixture unchanged when normalization is disabled."""
    frame = pl.read_parquet(real_fixture_paths[0])

    assert _apply_t_normalization(frame, None).equals(frame)


@pytest.mark.parametrize(
    "reference_range",
    [
        (1.0, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0,),
        "0,1",
    ],
)
def test_t_normalization_rejects_invalid_ranges(
    reference_range: object, real_fixture_paths: list[Path]
) -> None:
    """Reject reversed, nonfinite, and malformed t-normalization ranges."""
    frame = pl.read_parquet(real_fixture_paths[0])

    with pytest.raises(ValueError, match="t_normalization_range"):
        _apply_t_normalization(frame, reference_range)  # type: ignore[arg-type]


def test_t_normalization_rejects_empty_or_zero_mean_reference(
    real_fixture_paths: list[Path],
) -> None:
    """Reject empty references and zero or nonfinite reference means."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    zero_reference = frame.with_columns(
        pl.when(pl.col("Time") == pl.col("Time").min())
        .then(0.0)
        .otherwise(pl.col(wavelength))
        .alias(wavelength)
    )
    nonfinite_reference = frame.with_columns(
        pl.when(pl.col("Time") == pl.col("Time").min())
        .then(float("inf"))
        .otherwise(pl.col(wavelength))
        .alias(wavelength)
    )

    with pytest.raises(ValueError, match="reference interval"):
        _apply_t_normalization(frame, (-2.0, -1.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_t_normalization(zero_reference, (0.0, 0.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_t_normalization(nonfinite_reference, (0.0, 0.0))


def test_w_normalization_makes_real_fixture_reference_mean_one(
    real_fixture_paths: list[Path],
) -> None:
    """Normalize every row by its inclusive real-wavelength reference mean."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[1], wavelengths[-2])
    reference_columns = [
        column
        for column, wavelength in zip(wavelength_columns, wavelengths)
        if reference_range[0] <= wavelength <= reference_range[1]
    ]

    normalized = _apply_w_normalization(frame, reference_range)

    assert normalized.select(reference_columns).mean_horizontal().to_list() == (
        pytest.approx([1.0] * frame.height)
    )
    assert normalized.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert normalized.shape == frame.shape


def test_w_normalization_keeps_metadata_rows_separate(
    real_fixture_paths: list[Path],
) -> None:
    """Use a distinct wavelength reference mean for each metadata row."""
    frame = pl.read_parquet(real_fixture_paths[0]).head(2)
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[0], wavelengths[1])
    reference_columns = wavelength_columns[:2]

    normalized = _apply_w_normalization(frame, reference_range)

    for row_index in range(frame.height):
        divisor = frame.select(reference_columns).row(row_index)
        reference_mean = sum(divisor) / len(divisor)
        assert normalized.select(wavelength_columns).row(row_index) == pytest.approx(
            [
                value / reference_mean
                for value in frame.select(wavelength_columns).row(row_index)
            ]
        )


def test_w_normalization_none_preserves_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Return the validated real fixture unchanged when normalization is disabled."""
    frame = pl.read_parquet(real_fixture_paths[0])

    assert _apply_w_normalization(frame, None).equals(frame)


@pytest.mark.parametrize(
    "reference_range",
    [
        (1.0, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0,),
        "0,1",
    ],
)
def test_w_normalization_rejects_invalid_ranges(
    reference_range: object, real_fixture_paths: list[Path]
) -> None:
    """Reject reversed, nonfinite, and malformed w-normalization ranges."""
    frame = pl.read_parquet(real_fixture_paths[0])

    with pytest.raises(ValueError, match="w_normalization_range"):
        _apply_w_normalization(frame, reference_range)  # type: ignore[arg-type]


def test_w_normalization_rejects_empty_or_zero_mean_reference(
    real_fixture_paths: list[Path],
) -> None:
    """Reject empty references and zero or nonfinite per-row reference means."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[0], wavelengths[1])
    zero_reference = frame.with_columns(
        pl.lit(0.0).alias(column) for column in wavelength_columns[:2]
    )
    nonfinite_reference = frame.with_columns(
        pl.lit(float("inf")).alias(column) for column in wavelength_columns[:2]
    )

    with pytest.raises(ValueError, match="reference interval"):
        _apply_w_normalization(frame, (-2.0, -1.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_w_normalization(zero_reference, reference_range)
    with pytest.raises(ValueError, match="reference mean"):
        _apply_w_normalization(nonfinite_reference, reference_range)
