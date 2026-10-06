"""Tests for filtering, sorting, and summarizing cataloged files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.catalog import build_catalog_config, run_catalog
from flat_pca.webui.services.catalog_query import (
    FileQuery,
    category_options,
    existing_stems,
    list_files,
    metadata_warnings,
    parse_file_query,
)
from flat_pca.webui.services.catalog_store import apply_catalog_result
from flat_pca.webui.settings import Settings


@pytest.fixture
def cataloged(settings: Settings, database: Database, tmp_path: Path) -> Database:
    """Return the database after one catalog run over the fixtures."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = build_catalog_config(settings, {})
    run_catalog(config, run_dir)
    apply_catalog_result(database, config, run_dir)
    return database


def _stems(rows: list[dict[str, object]]) -> list[object]:
    """Return the stems of file rows."""
    return [row["stem"] for row in rows]


def test_parse_file_query_reads_filters_and_sort(settings: Settings) -> None:
    """Typed filters and the sort order are parsed from query parameters."""
    query = parse_file_query(
        {
            "q": " run ",
            "eq__lot": "A",
            "min__yield_pct": "90",
            "max__yield_pct": "",
            "max__date": "2026-02-01T00:00",
            "sort": "date",
            "order": "desc",
        },
        settings.metadata_columns,
    )

    assert query == FileQuery(
        text="run",
        equals={"lot": "A"},
        ranges={
            "yield_pct": (90.0, None),
            "date": (None, datetime.fromisoformat("2026-02-01")),
        },
        sort_by="date",
        descending=True,
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"eq__unknown": "x"}, "unknown metadata column"),
        ({"min__lot": "1"}, "does not fit"),
        ({"eq__yield_pct": "1"}, "does not fit"),
        ({"min__yield_pct": "abc"}, "invalid bound"),
        ({"sort": "path"}, "unknown sort column"),
        ({"order": "up"}, "invalid sort order"),
    ],
)
def test_parse_file_query_rejects_invalid_parameters(
    settings: Settings, parameters: dict[str, str], message: str
) -> None:
    """Unknown columns, mismatched filters, and bad values are rejected."""
    with pytest.raises(ValueError, match=message):
        parse_file_query(parameters, settings.metadata_columns)


def test_list_files_uses_natural_order_and_joins_metadata(cataloged: Database) -> None:
    """Stems sort naturally and files without CSV rows have null metadata."""
    rows = list_files(cataloged, FileQuery())

    assert _stems(rows) == ["run-1", "run-2", "run-10"]
    assert rows[0]["lot"] == "A"
    assert rows[0]["n_steps"] == 2
    assert rows[0]["n_segments"] == 3
    assert rows[2]["lot"] is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (FileQuery(descending=True), ["run-10", "run-2", "run-1"]),
        (FileQuery(text="RUN-1"), ["run-1", "run-10"]),
        (FileQuery(equals={"lot": "B"}), ["run-2"]),
        (FileQuery(ranges={"yield_pct": (90.0, None)}), ["run-1"]),
        (
            FileQuery(ranges={"date": (datetime.fromisoformat("2026-02-01"), None)}),
            ["run-2"],
        ),
        (FileQuery(sort_by="yield_pct"), ["run-2", "run-1", "run-10"]),
        (FileQuery(sort_by="yield_pct", descending=True), ["run-1", "run-2", "run-10"]),
    ],
)
def test_list_files_filters_and_sorts(
    cataloged: Database, query: FileQuery, expected: list[str]
) -> None:
    """Filters combine with AND; null sort values come last."""
    assert _stems(list_files(cataloged, query)) == expected


def test_category_options_and_metadata_warnings(cataloged: Database) -> None:
    """Category values and CSV/file mismatches are reported."""
    warnings = metadata_warnings(cataloged)

    assert category_options(cataloged) == {"lot": ["A", "B"]}
    assert warnings.files_without_metadata == ["run-10"]
    assert warnings.metadata_without_files == ["ghost"]


def test_existing_stems_drops_unknown_stems(cataloged: Database) -> None:
    """Only cataloged stems are kept, once each."""
    assert existing_stems(cataloged, ["run-10", "ghost", "run-1", "run-1"]) == [
        "run-1",
        "run-10",
    ]
    assert existing_stems(cataloged, []) == []
