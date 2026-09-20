"""Tests for edge trimming."""

from pathlib import Path

import polars as pl

from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns,
    apply_edge_trim,
)

_META_COLUMNS = {"Time", "Step", "Sequence", "StepTime", "ReverseStepTime"}


def _fixture_with_step_time(source_path: Path) -> pl.DataFrame:
    """Load a real fixture and add its edge-trimming time columns.

    Parameters
    ----------
    source_path : Path
        Real fixture Parquet path.

    Returns
    -------
    pl.DataFrame
        Fixture data with ``StepTime`` and ``ReverseStepTime`` columns.
    """
    return add_step_time_columns(pl.scan_parquet(source_path)).collect()


def test_apply_edge_trim_drops_rows_near_segment_edges_by_default(
    real_fixture_paths: list[Path],
) -> None:
    """Drop rows within either trimmed edge by default."""
    frame = _fixture_with_step_time(real_fixture_paths[0])

    trimmed = apply_edge_trim(frame, [5000.0, 5000.0])
    assert isinstance(trimmed, pl.DataFrame)

    assert trimmed.height < frame.height
    assert (
        trimmed.filter(
            (pl.col("StepTime") < 5000.0)
            | (pl.col("ReverseStepTime") < 5000.0)
        ).height
        == 0
    )


def test_apply_edge_trim_nulls_non_meta_columns_when_requested(
    real_fixture_paths: list[Path],
) -> None:
    """Null spectral values within either trimmed edge when requested."""
    frame = _fixture_with_step_time(real_fixture_paths[0])
    wavelength_columns = [
        column for column in frame.columns if column not in _META_COLUMNS
    ]

    trimmed = apply_edge_trim(frame, [5000.0, 5000.0], action="null")
    assert isinstance(trimmed, pl.DataFrame)

    step_time = trimmed.get_column("StepTime").to_list()
    reverse_step_time = trimmed.get_column("ReverseStepTime").to_list()
    sample_values = trimmed.get_column(wavelength_columns[0]).to_list()

    for row_step_time, row_reverse_step_time, value in zip(
        step_time, reverse_step_time, sample_values, strict=True
    ):
        if row_step_time < 5000.0 or row_reverse_step_time < 5000.0:
            assert value is None
        else:
            assert value is not None

    for column in _META_COLUMNS:
        assert trimmed.get_column(column).null_count() == 0

    unchanged = apply_edge_trim(frame, None)
    assert unchanged.equals(frame)
