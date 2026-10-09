"""Tests for the catalog's file table: its settings and its rows."""

from __future__ import annotations

import polars as pl
import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.datatable import apply_state, parse_state
from flat_pca.webui.services.file_table import file_frame, file_table_config
from flat_pca.webui.settings import Settings


def test_file_frame_types_every_column(cataloged: Database) -> None:
    """The frame holds the files in natural order with typed metadata."""
    frame = file_frame(cataloged)

    assert frame["stem"].to_list() == ["run-1", "run-2", "run-10"]
    assert frame.schema["lot"] == pl.String
    assert frame.schema["yield_pct"] == pl.Float64
    assert frame.schema["date"] == pl.Datetime("us")
    assert frame.schema["n_rows"] == pl.Int64


def test_file_frame_without_files_keeps_its_columns(database: Database) -> None:
    """An empty catalog gives an empty frame with every column."""
    frame = file_frame(database)

    assert frame.height == 0
    assert frame.columns == [
        "stem", "path", "n_rows", "n_steps", "n_segments", "lot", "date", "yield_pct"
    ]


def test_file_table_config_filters_metadata_but_not_statistics(settings: Settings) -> None:
    """The stem and metadata columns have filters; the statistics do not."""
    config = file_table_config(settings.metadata_columns)

    assert {name: column.filter for name, column in config.filtered_columns.items()} == {
        "stem": "text",
        "lot": "choice",
        "date": "datetime",
        "yield_pct": "number",
    }
    assert list(config.sortable_columns) == [
        "stem", "lot", "date", "yield_pct", "n_steps", "n_segments", "n_rows"
    ]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"files.order": ["desc"]}, ["run-10", "run-2", "run-1"]),
        ({"files.q__stem": ["RUN-1"]}, ["run-1", "run-10"]),
        ({"files.eq__lot": ["B"]}, ["run-2"]),
        ({"files.min__yield_pct": ["90"]}, ["run-1"]),
        ({"files.min__date": ["2026-02-01T00:00"]}, ["run-2"]),
        ({"files.sort": ["yield_pct"]}, ["run-2", "run-1", "run-10"]),
        (
            {"files.sort": ["yield_pct"], "files.order": ["desc"]},
            ["run-1", "run-2", "run-10"],
        ),
    ],
)
def test_file_table_filters_and_sorts(
    cataloged: Database, settings: Settings, parameters: dict[str, list[str]], expected: list[str]
) -> None:
    """Filters combine with AND; stems sort naturally; null sort values come last."""
    config = file_table_config(settings.metadata_columns)

    view = apply_state(file_frame(cataloged), parse_state(parameters, config), config)

    assert [row["stem"] for row in view.rows] == expected
    assert view.matching_keys == expected
