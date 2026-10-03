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


@pytest.fixture
def indexed_flattened() -> pl.DataFrame:
    """Return flattened features with metadata columns, including a list column."""
    return pl.DataFrame(
        {
            "label": ["x", "y"],
            "source": ["a", "b"],
            "650.0nm_1_1_0.00": [1.0, None],
            "651.0nm_1_1_0.00": [2.0, None],
            "lot": [10, 20],
            "650.0nm_1_1_0.50": [None, 3.0],
            "651.0nm_1_1_0.50": [None, 4.0],
            "tags": [["p"], ["q", "r"]],
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_flatten_to_long_retains_index_columns(
    indexed_flattened: pl.DataFrame, lazy: bool
) -> None:
    """Repeat each input row's index values on its long rows in the given order."""
    frame = indexed_flattened.lazy() if lazy else indexed_flattened
    result = flatten_to_long(frame, index=["lot", "tags", "label"])
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert result.columns == [
        "source", "lot", "tags", "label",
        "Step", "Sequence", "StepTime", "wavelength", "intensity",
    ]
    assert result.height == 8
    assert result.filter(pl.col("source") == "a")["lot"].to_list() == [10] * 4
    assert result.filter(pl.col("source") == "b")["label"].to_list() == ["y"] * 4
    assert result.filter(pl.col("source") == "b")["tags"].to_list() == [["q", "r"]] * 4


@pytest.mark.parametrize("lazy", [False, True])
def test_flatten_to_wide_retains_index_columns(
    indexed_flattened: pl.DataFrame, lazy: bool
) -> None:
    """Keep ungroupable index columns on every wide grid row in the given order."""
    frame = indexed_flattened.lazy() if lazy else indexed_flattened
    result = flatten_to_wide(frame, index=["tags", "lot", "label"])
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    expected = pl.DataFrame(
        {
            "source": ["a", "a", "b", "b"],
            "tags": [["p"], ["p"], ["q", "r"], ["q", "r"]],
            "lot": [10, 10, 20, 20],
            "label": ["x", "x", "y", "y"],
            "Step": [1, 1, 1, 1],
            "Sequence": [1, 1, 1, 1],
            "StepTime": [0.0, 0.5, 0.0, 0.5],
            "650.0nm": [1.0, None, None, 3.0],
            "651.0nm": [2.0, None, None, 4.0],
        }
    )
    assert result.equals(expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_flatten_to_wide_drops_null_rows_ignoring_index(
    indexed_flattened: pl.DataFrame, lazy: bool
) -> None:
    """Drop grid rows by wavelength values only, even with non-null index values."""
    flattened = indexed_flattened.drop("label", "tags")
    frame = flattened.lazy() if lazy else flattened
    result = flatten_to_wide(frame, index="lot", drop_all_null_rows=True)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    expected = pl.DataFrame(
        {
            "source": ["a", "b"],
            "lot": [10, 20],
            "Step": [1, 1],
            "Sequence": [1, 1],
            "StepTime": [0.0, 0.5],
            "650.0nm": [1.0, 3.0],
            "651.0nm": [2.0, 4.0],
        }
    )
    assert result.equals(expected)


@pytest.mark.parametrize(
    ("index", "match"),
    [
        (["lot", "lot"], "must be unique"),
        (["missing"], "not found"),
        ("source", "conflict"),
        ("Step", "conflict"),
        ("Sequence", "conflict"),
        ("StepTime", "conflict"),
        ("wavelength", "conflict"),
        ("__flatten_source_row", "conflict"),
        ("__flatten_feature", "conflict"),
        ("__flatten_value", "conflict"),
    ],
)
def test_unflatten_rejects_invalid_index(index: str | list[str], match: str) -> None:
    """Reject duplicated, missing, and reserved index column names."""
    frame = pl.DataFrame(
        {
            "source": ["a"],
            "650.0nm_1_1_0.00": [1.0],
            "lot": [1],
            "Step": [1],
            "Sequence": [1],
            "StepTime": [0.0],
            "wavelength": [650.0],
            "__flatten_source_row": [0],
            "__flatten_feature": ["f"],
            "__flatten_value": [0.0],
        }
    )
    with pytest.raises(ValueError, match=match):
        flatten_to_long(frame, index=index)
    with pytest.raises(ValueError, match=match):
        flatten_to_wide(frame.lazy(), index=index)


def test_flatten_to_long_rejects_index_conflicting_with_value_name() -> None:
    """Reject an index column with the same name as the long value column."""
    frame = pl.DataFrame({"source": ["a"], "650.0nm_1_1_0.00": [1.0], "value": [1]})
    with pytest.raises(ValueError, match="conflict"):
        flatten_to_long(frame, value_name="value", index="value")
    assert flatten_to_wide(frame, index="value").columns[1] == "value"


def test_flatten_to_wide_rejects_index_conflicting_with_wavelength() -> None:
    """Reject an index column with the same name as an output wavelength column."""
    frame = pl.DataFrame({"source": ["a"], "650.0nm_1_1_0.00": [1.0], "650.0nm": [1]})
    with pytest.raises(ValueError, match="conflict"):
        flatten_to_wide(frame, index="650.0nm")
    assert flatten_to_long(frame, index="650.0nm").columns[1] == "650.0nm"


def test_unflatten_rejects_columns_outside_index() -> None:
    """Keep rejecting non-feature columns that are not listed in index."""
    frame = pl.DataFrame(
        {"source": ["a"], "650.0nm_1_1_0.00": [1.0], "lot": [1], "label": ["x"]}
    )
    with pytest.raises(ValueError, match="uniquely decode"):
        flatten_to_long(frame, index="lot")
    with pytest.raises(ValueError, match="uniquely decode"):
        flatten_to_wide(frame.lazy(), index=["label"])
