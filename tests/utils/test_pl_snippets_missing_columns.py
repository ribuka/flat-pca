"""Tests for missing-value column pruning in the polars utilities."""

import math

import polars as pl
import pytest

from flat_pca.utils import drop_all_null_columns_from_polars


@pytest.fixture
def mixed_missing_frame() -> pl.DataFrame:
    """Build a 4-row frame mixing nulls and NaNs across dtypes.

    Returns
    -------
    pl.DataFrame
        Missing ratios counting NaN / ignoring NaN: ``f64`` 0.5 / 0.25,
        ``nan_heavy`` 0.75 / 0.0, ``f32`` 0.25 / 0.0, ``int`` 0.25 / 0.25,
        ``text`` 0.5 / 0.5, and ``dense`` 0.0 / 0.0.
    """
    return pl.DataFrame(
        {
            "f64": [1.0, math.nan, None, 4.0],
            "nan_heavy": [math.nan, math.nan, math.nan, 1.0],
            "f32": pl.Series([math.nan, 1.0, 2.0, 3.0], dtype=pl.Float32),
            "int": [1, None, 3, 4],
            "text": ["NaN", None, None, "x"],
            "dense": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.mark.parametrize(
    ("include_nan_missing", "threshold", "expected"),
    [
        (True, 0.0, ["dense"]),
        (True, 0.25, ["f32", "int", "dense"]),
        (True, 0.5, ["f64", "f32", "int", "text", "dense"]),
        (True, 1.0, ["f64", "nan_heavy", "f32", "int", "text", "dense"]),
        (False, 0.0, ["nan_heavy", "f32", "dense"]),
        (False, 0.25, ["f64", "nan_heavy", "f32", "int", "dense"]),
        (False, 1.0, ["f64", "nan_heavy", "f32", "int", "text", "dense"]),
    ],
)
def test_drop_all_null_columns_keeps_columns_at_or_below_threshold(
    mixed_missing_frame: pl.DataFrame,
    include_nan_missing: bool,
    threshold: float,
    expected: list[str],
) -> None:
    """Keep columns whose missing ratio is at most the threshold, in order."""
    result = drop_all_null_columns_from_polars(
        mixed_missing_frame,
        include_nan_missing=include_nan_missing,
        threshold=threshold,
    )

    assert isinstance(result, pl.DataFrame)
    assert result.columns == expected
    assert result.equals(mixed_missing_frame.select(expected))


def test_drop_all_null_columns_ignores_nan_text_in_non_float_columns() -> None:
    """Count only nulls for integer and string columns."""
    frame = pl.DataFrame({"int": [1, 2], "text": ["NaN", "nan"]})

    result = drop_all_null_columns_from_polars(frame, threshold=0.0)

    assert result.columns == ["int", "text"]


def test_drop_all_null_columns_keeps_lazy_frames_lazy(
    mixed_missing_frame: pl.DataFrame,
) -> None:
    """Return a LazyFrame matching the eager result for LazyFrame input."""
    result = drop_all_null_columns_from_polars(
        mixed_missing_frame.lazy(), threshold=0.25
    )

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().equals(
        mixed_missing_frame.select("f32", "int", "dense")
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_drop_all_null_columns_returns_empty_frame_unchanged(lazy: bool) -> None:
    """Return a frame without rows as is, keeping every column."""
    frame = pl.DataFrame(schema={"f64": pl.Float64, "text": pl.String})
    source = frame.lazy() if lazy else frame

    result = drop_all_null_columns_from_polars(source, threshold=0.0)

    assert result is source


def test_drop_all_null_columns_handles_frames_without_columns() -> None:
    """Accept a frame that has no columns."""
    frame = pl.DataFrame()

    result = drop_all_null_columns_from_polars(frame)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (0, 0)


def test_drop_all_null_columns_handles_frames_without_float_columns() -> None:
    """Count missing values when no column is floating point."""
    frame = pl.DataFrame({"int": [1, None], "text": ["a", "b"]})

    result = drop_all_null_columns_from_polars(frame, threshold=0.4)

    assert result.columns == ["text"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_drop_all_null_columns_rejects_out_of_range_threshold(
    mixed_missing_frame: pl.DataFrame,
    threshold: float,
) -> None:
    """Reject thresholds outside 0.0-1.0."""
    with pytest.raises(ValueError, match="threshold must be between"):
        drop_all_null_columns_from_polars(mixed_missing_frame, threshold=threshold)
