"""Tests for the table of the files behind the points of a figure."""

from __future__ import annotations

import polars as pl
import pytest

from flat_pca.webui.services.point_table import point_table


def test_rows_hold_the_stem_metadata_and_extra_columns() -> None:
    """Each row is the stem, the metadata, and the extra values, formatted."""
    samples = pl.DataFrame(
        {
            "stem": ["b", "a"],
            "lot": ["B", None],
            "date": ["2024-01-02T03:04:05", None],
        }
    ).with_columns(pl.col("date").str.to_datetime())

    table = point_table(samples, {"T²": [1.23456789, 2.0], "flag": ["exceeds", ""]})

    assert table.columns == ["file", "lot", "date", "T²", "flag"]
    assert table.rows == [
        ["b", "B", "2024-01-02 03:04:05", "1.23457", "exceeds"],
        ["a", "", "", "2", ""],
    ]


def test_metadata_named_like_an_extra_column_is_kept() -> None:
    """A metadata column may share its name with an extra column."""
    samples = pl.DataFrame({"stem": ["a"], "PC1": ["meta"]})

    table = point_table(samples, {"PC1": [0.5]})

    assert table.columns == ["file", "PC1", "PC1"]
    assert table.rows == [["a", "meta", "0.5"]]


def test_extra_column_of_another_length_is_rejected() -> None:
    """An extra column must hold one value per file."""
    samples = pl.DataFrame({"stem": ["a", "b"]})

    with pytest.raises(ValueError, match="2 values"):
        point_table(samples, {"Q": [1.0]})
