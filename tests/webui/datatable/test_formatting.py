"""Tests for the display text of cell values and column types."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from flat_pca.webui.datatable import dtype_label, format_value


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.String(), "str"),
        (pl.Categorical(), "cat"),
        (pl.Enum(["a", "b"]), "enum"),
        (pl.Boolean(), "bool"),
        (pl.Int8(), "i8"),
        (pl.Int64(), "i64"),
        (pl.UInt32(), "u32"),
        (pl.Float32(), "f32"),
        (pl.Float64(), "f64"),
        (pl.Date(), "date"),
        (pl.Time(), "time"),
        (pl.Binary(), "binary"),
        (pl.Null(), "null"),
        (pl.Datetime("us"), "datetime[μs]"),
        (pl.Datetime("ns", "UTC"), "datetime[ns, UTC]"),
        (pl.Duration("ms"), "duration[ms]"),
        (pl.Decimal(10, 2), "decimal[10,2]"),
        (pl.List(pl.Int64), "list[i64]"),
        (pl.Array(pl.Float64, 3), "array[f64, 3]"),
        (pl.Array(pl.Int64, (2, 3)), "array[i64, (2, 3)]"),
        (pl.Struct({"a": pl.Int64, "b": pl.String}), "struct[2]"),
    ],
)
def test_dtype_label_gives_the_short_polars_name(dtype: pl.DataType, expected: str) -> None:
    """Types are named as polars prints them over a frame's columns."""
    assert dtype_label(dtype) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (datetime.fromisoformat("2026-01-02T03:04:05"), "2026-01-02 03:04:05"),
        (1 / 3, "0.333333"),
        (12, "12"),
        ("x", "x"),
    ],
)
def test_format_value_formats_cells(value: object, expected: str) -> None:
    """Cells show blanks, datetimes to the second, and floats to six digits."""
    assert format_value(value) == expected
