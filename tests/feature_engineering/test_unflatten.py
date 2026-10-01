"""Tests for restoring flattened spectral coordinates."""

from pathlib import Path

import polars as pl
import pytest

from flat_pca import flatten_to_long, flatten_to_wide
from flat_pca.feature_engineering.flatten_pca.flatten import flatten_inputs
from flat_pca.feature_engineering.flatten_pca.input import load_and_validate_inputs
from flat_pca.feature_engineering.preprocess import add_step_time_columns
from flat_pca.visualize import create_heatmap


@pytest.mark.parametrize("lazy", [False, True])
def test_flatten_to_wide_restores_real_spectral_values(
    real_fixture_paths: list[Path], lazy: bool
) -> None:
    """Round trip a real fixture while omitting irrecoverable time metadata."""
    path = real_fixture_paths[0]
    original = add_step_time_columns(load_and_validate_inputs([path])[0][1]).collect()
    flattened = flatten_inputs([(path, original.lazy() if lazy else original)])

    restored = flatten_to_wide(flattened)
    assert isinstance(restored, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(restored, pl.LazyFrame):
        restored = restored.collect()
    spectral = [column for column in original.columns if column.endswith("nm")]
    expected = (
        original.select("Step", "Sequence", "StepTime", *spectral)
        .with_columns(pl.lit(path.as_posix()).alias("source"))
        .select("source", "Step", "Sequence", "StepTime", *spectral)
        .sort("source", "Step", "Sequence", "StepTime")
    )
    assert restored.columns == expected.columns
    assert restored.equals(expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_flatten_to_long_feeds_heatmap(lazy: bool) -> None:
    """Preserve typed coordinates and feed the explicit long heatmap path."""
    flattened = pl.DataFrame(
        {
            "source": ["a"],
            "651.0nm_1_1_0.50": [4.0],
            "650.0nm_1_1_0.00": [1.0],
            "651.0nm_1_1_0.00": [2.0],
            "650.0nm_1_1_0.50": [3.0],
        }
    )
    result = flatten_to_long(flattened.lazy() if lazy else flattened)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert result.schema == {
        "source": pl.String,
        "Step": pl.Int64,
        "Sequence": pl.Int64,
        "StepTime": pl.Float64,
        "wavelength": pl.Float64,
        "intensity": pl.Float64,
    }
    figure = create_heatmap(
        result.filter((pl.col("source") == "a") & (pl.col("Step") == 1)),
        y_name="StepTime",
    )
    assert list(figure.data[0].x) == [650.0, 651.0]
    assert figure.data[0].z.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("lazy", [False, True])
def test_unflatten_preserves_missing_grid_values(lazy: bool) -> None:
    """Keep null long values and optionally remove empty wide grid rows."""
    flattened = pl.DataFrame(
        {
            "source": ["a", "b"],
            "650.0nm_1_1_0.00": [1.0, None],
            "651.0nm_1_1_0.00": [2.0, None],
            "650.0nm_1_1_0.50": [None, 3.0],
            "651.0nm_1_1_0.50": [None, 4.0],
        }
    )
    frame = flattened.lazy() if lazy else flattened
    long = flatten_to_long(frame, value_name="value")
    wide = flatten_to_wide(frame)
    dropped = flatten_to_wide(frame, drop_all_null_rows=True)
    if lazy:
        long, wide, dropped = long.collect(), wide.collect(), dropped.collect()
    assert long.height == 8
    assert long["value"].null_count() == 4
    assert wide.height == 4
    assert dropped.height == 2
    assert dropped["source"].to_list() == ["a", "b"]
    assert dropped.columns == [
        "source", "Step", "Sequence", "StepTime", "650.0nm", "651.0nm"
    ]


@pytest.mark.parametrize("name", ["score", "650nm_01_1_0.00", "650.0nm_1_1_0.0"])
def test_unflatten_rejects_noncanonical_columns(name: str) -> None:
    """Reject arbitrary metadata and ambiguous feature spellings."""
    frame = pl.DataFrame({"source": ["a"], name: [1.0]})
    with pytest.raises(ValueError, match="uniquely decode"):
        flatten_to_long(frame)
    with pytest.raises(ValueError, match="uniquely decode"):
        flatten_to_wide(frame.lazy())
