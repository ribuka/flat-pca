"""Tests for the columns that can color the score and T² / Q figures."""

from __future__ import annotations

import polars as pl

from flat_pca.webui.services.scored_samples import choose_metadata_column, color_columns
from flat_pca.webui.services.scores import trajectory_color_values


def _samples() -> pl.DataFrame:
    """Return ``samples.parquet`` of three files with two metadata columns."""
    return pl.DataFrame(
        {
            "source": ["a.parquet", "b.parquet", "c.parquet"],
            "stem": ["a", "b", "c"],
            "lot": ["L1", None, "L2"],
            "yield_pct": [1.0, 2.0, 3.0],
        }
    )


def test_color_columns_put_the_file_name_first() -> None:
    """The file name comes before the metadata columns, and the source is left out."""
    assert color_columns(_samples()) == ["stem", "lot", "yield_pct"]


def test_file_name_can_be_chosen_and_be_the_default() -> None:
    """``stem`` is chosen like a metadata column, also as the default."""
    options = color_columns(_samples())

    assert choose_metadata_column("stem", "lot", options) == "stem"
    assert choose_metadata_column(None, "stem", options) == "stem"
    assert choose_metadata_column("missing", "stem", options) == "stem"
    assert choose_metadata_column("", "stem", options) is None


def test_trajectory_color_values_follow_the_trajectory_order() -> None:
    """The values are those of the given stems in their order, named by the column."""
    values = trajectory_color_values(_samples(), ["c", "a", "b"], "lot")

    assert values.name == "lot"
    assert values.to_list() == ["L2", "L1", None]
    assert trajectory_color_values(_samples(), ["b"], "stem").to_list() == ["b"]
