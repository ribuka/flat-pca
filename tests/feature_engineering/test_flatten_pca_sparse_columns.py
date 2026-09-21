"""Tests for sparse feature-column pruning after Flatten-PCA flattening."""

from pathlib import Path

import polars as pl

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


def _combo_feature_name(
    canonical: pl.DataFrame, wavelength: str, time_value: float
) -> str:
    """Format the feature name for the combination present at a Time value.

    Parameters
    ----------
    canonical : pl.DataFrame
        Frame with ``StepTime`` already generated, containing ``time_value``.
    wavelength : str
        Wavelength column name to format the feature name for.
    time_value : float
        ``Time`` value identifying the row to read Step/Sequence/StepTime from.

    Returns
    -------
    str
        The formatted flattened feature name for that combination.
    """
    row = (
        canonical.filter(pl.col("Time") == time_value)
        .select("Step", "Sequence", "StepTime")
        .row(0, named=True)
    )
    return (
        f"{wavelength}_{int(row['Step'])}_{int(row['Sequence'])}"
        f"_{float(row['StepTime']):.2f}"
    )


def _build_mixed_sparsity_flattened(
    real_fixture_paths: list[Path],
) -> tuple[pl.LazyFrame, str, str]:
    """Flatten 4 real-fixture-derived variants with distinct null ratios.

    Every variant keeps row 0 (the minimum ``Time``), so
    ``add_step_time_columns``'s per-segment ``StepTime`` origin never
    shifts between variants. One combination (the file's last row) is
    missing from 2 of 4 variants (50% null); another (the second-to-last
    row) is missing from 1 of 4 (25% null).

    Parameters
    ----------
    real_fixture_paths : list[Path]
        Real fixture Parquet paths; only the first file's data is used,
        paired with 4 distinct path labels.

    Returns
    -------
    tuple[pl.LazyFrame, str, str]
        Flattened frame, the 25%-missing column name, and the 50%-missing
        column name.
    """
    frame = pl.read_parquet(real_fixture_paths[0])
    canonical = _add_step_time_columns(frame)
    assert isinstance(canonical, pl.DataFrame)
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    last_time = frame["Time"][frame.height - 1]
    second_last_time = frame["Time"][frame.height - 2]
    half_missing_column = _combo_feature_name(canonical, wavelength, last_time)
    quarter_missing_column = _combo_feature_name(
        canonical, wavelength, second_last_time
    )

    variant_frames = [
        frame,
        frame.filter(pl.col("Time") != last_time),
        frame.filter(pl.col("Time") != last_time),
        frame.filter(pl.col("Time") != second_last_time),
    ]
    inputs = [
        (path, _add_step_time_columns(variant).lazy())
        for path, variant in zip(real_fixture_paths[:4], variant_frames, strict=True)
    ]

    flattened = _flatten_inputs(inputs)
    assert isinstance(flattened, pl.LazyFrame)
    return flattened, quarter_missing_column, half_missing_column


def test_drop_sparse_feature_columns_keeps_columns_at_or_below_threshold(
    real_fixture_paths: list[Path],
) -> None:
    """Keep a column whose null ratio exactly equals max_null_ratio."""
    flattened, quarter_missing_column, half_missing_column = (
        _build_mixed_sparsity_flattened(real_fixture_paths)
    )

    kept = _drop_sparse_feature_columns(flattened, max_null_ratio=0.25).collect()

    assert quarter_missing_column in kept.columns
    assert half_missing_column not in kept.columns


def test_drop_sparse_feature_columns_drops_columns_above_threshold(
    real_fixture_paths: list[Path],
) -> None:
    """Drop a column whose null ratio is strictly above max_null_ratio."""
    flattened, quarter_missing_column, _ = _build_mixed_sparsity_flattened(
        real_fixture_paths
    )

    dropped = _drop_sparse_feature_columns(flattened, max_null_ratio=0.24).collect()

    assert quarter_missing_column not in dropped.columns


def test_drop_sparse_feature_columns_prunes_a_real_materialized_frame(
    real_fixture_paths: list[Path],
) -> None:
    """Prune a materialized frame derived from real Parquet fixtures."""
    flattened, quarter_missing_column, half_missing_column = (
        _build_mixed_sparsity_flattened(real_fixture_paths)
    )

    result = _drop_sparse_feature_columns(
        flattened.collect(),
        max_null_ratio=0.25,
    )

    assert isinstance(result, pl.DataFrame)
    assert quarter_missing_column in result.columns
    assert half_missing_column not in result.columns


def test_drop_sparse_feature_columns_always_keeps_source(
    real_fixture_paths: list[Path],
) -> None:
    """Never drop the source column, even with the strictest threshold."""
    flattened, _, _ = _build_mixed_sparsity_flattened(real_fixture_paths)

    result = _drop_sparse_feature_columns(flattened, max_null_ratio=0.0).collect()

    assert "source" in result.columns
