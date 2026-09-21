"""Tests for the wavelength-range column filter."""

from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering import flatten_pca, preprocess_and_flatten
from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns,
)
from flat_pca.feature_engineering.preprocess import (
    apply_wavelength_range_filter as _apply_wavelength_range_filter,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence"}
STEP_TIME_COLUMNS = ["StepTime", "ReverseStepTime"]


def test_apply_wavelength_range_filter_keeps_only_inclusive_range_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Keep only real-fixture wavelength columns within an inclusive interval."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = sorted(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )
    reference_range = (wavelengths[1], wavelengths[-2])
    expected_columns = [
        column
        for column in wavelength_columns
        if reference_range[0] <= float(column.removesuffix("nm")) <= reference_range[1]
    ]

    filtered = _apply_wavelength_range_filter(frame, reference_range)

    assert filtered.columns == ["Time", "Step", "Sequence", *expected_columns]
    assert filtered.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert filtered.select(expected_columns).equals(frame.select(expected_columns))
    assert expected_columns != wavelength_columns


def test_apply_wavelength_range_filter_keeps_boundary_wavelengths(
    real_fixture_paths: list[Path],
) -> None:
    """Keep wavelength columns exactly at the lower and upper bound (inclusive)."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = sorted(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )
    reference_range = (wavelengths[0], wavelengths[-1])

    filtered = _apply_wavelength_range_filter(frame, reference_range)

    assert set(filtered.columns[3:]) == set(wavelength_columns)


def test_apply_wavelength_range_filter_keeps_step_time_columns_when_present(
    real_fixture_paths: list[Path],
) -> None:
    """Preserve StepTime and ReverseStepTime columns produced upstream."""
    frame = add_step_time_columns(pl.read_parquet(real_fixture_paths[0]))
    wavelength_columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and column not in STEP_TIME_COLUMNS
    ]
    wavelengths = sorted(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )
    reference_range = (wavelengths[0], wavelengths[len(wavelengths) // 2])

    filtered = _apply_wavelength_range_filter(frame, reference_range)

    assert set(STEP_TIME_COLUMNS).issubset(filtered.columns)
    assert filtered.select(STEP_TIME_COLUMNS).equals(frame.select(STEP_TIME_COLUMNS))


def test_apply_wavelength_range_filter_none_preserves_real_fixture(
    real_fixture_paths: list[Path],
) -> None:
    """Return the validated real fixture unchanged when the filter is disabled."""
    frame = pl.read_parquet(real_fixture_paths[0])

    assert _apply_wavelength_range_filter(frame, None).equals(frame)


@pytest.mark.parametrize(
    "wavelength_range",
    [
        (800.0, 300.0),
        (300.0, float("nan")),
        (300.0, float("inf")),
        (300.0,),
        "300,800",
        (True, 800.0),
    ],
)
def test_apply_wavelength_range_filter_rejects_invalid_ranges(
    wavelength_range: object, real_fixture_paths: list[Path]
) -> None:
    """Reject reversed, nonfinite, boolean, and malformed wavelength ranges."""
    frame = pl.read_parquet(real_fixture_paths[0])

    with pytest.raises(ValueError, match="wavelength_range"):
        _apply_wavelength_range_filter(frame, wavelength_range)  # type: ignore[arg-type]


def test_apply_wavelength_range_filter_rejects_range_matching_no_wavelength_column(
    real_fixture_paths: list[Path],
) -> None:
    """Reject an interval that lies entirely outside every real wavelength."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    max_wavelength = max(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )

    with pytest.raises(ValueError, match="wavelength_range"):
        _apply_wavelength_range_filter(
            frame, (max_wavelength + 1000.0, max_wavelength + 2000.0)
        )


def test_preprocess_and_flatten_applies_wavelength_range_before_other_stages(
    real_fixture_paths: list[Path],
) -> None:
    """Restrict flattened features to the real-fixture wavelengths in range."""
    paths = real_fixture_paths[:3]
    first_frame = pl.read_parquet(paths[0])
    wavelength_columns = [
        column for column in first_frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = sorted(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )
    reference_range = (wavelengths[1], wavelengths[-2])

    prepared_inputs = []
    for path in sorted(paths):
        with_step_time = add_step_time_columns(pl.read_parquet(path))
        filtered = _apply_wavelength_range_filter(with_step_time, reference_range)
        prepared_inputs.append((path.resolve(), filtered))
    expected = _flatten_inputs(prepared_inputs)
    assert isinstance(expected, pl.DataFrame)

    flattened = preprocess_and_flatten(
        paths, wavelength_range=reference_range
    ).collect()

    assert flattened.equals(expected)
    kept_feature_wavelengths = {
        float(column.split("_", 1)[0].removesuffix("nm"))
        for column in flattened.columns[1:]
    }
    assert all(
        reference_range[0] <= wavelength <= reference_range[1]
        for wavelength in kept_feature_wavelengths
    )
    assert kept_feature_wavelengths < set(wavelengths)


def test_flatten_pca_fits_using_wavelength_range_filtered_features(
    real_fixture_paths: list[Path],
) -> None:
    """Fit PCA on flattened features restricted to the requested wavelength range."""
    paths = real_fixture_paths[:3]
    first_frame = pl.read_parquet(paths[0])
    wavelength_columns = [
        column for column in first_frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = sorted(
        float(column.removesuffix("nm")) for column in wavelength_columns
    )
    reference_range = (wavelengths[0], wavelengths[len(wavelengths) // 2])

    flattened = preprocess_and_flatten(paths, wavelength_range=reference_range)
    pca = flatten_pca(paths, n_component=2, wavelength_range=reference_range)

    assert pca.pca.n_features_in_ == flattened.collect().width - 1


def test_preprocess_and_flatten_rejects_invalid_wavelength_range(
    real_fixture_paths: list[Path],
) -> None:
    """Reject a malformed wavelength_range through the public API."""
    with pytest.raises(ValueError, match="wavelength_range"):
        preprocess_and_flatten(real_fixture_paths[:1], wavelength_range=(1.0, 0.0))
