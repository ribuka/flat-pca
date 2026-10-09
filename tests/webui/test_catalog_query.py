"""Tests for listing and summarizing cataloged files."""

from __future__ import annotations

from flat_pca.webui.database import Database
from flat_pca.webui.services.catalog_query import (
    category_options,
    existing_stems,
    list_files,
    metadata_warnings,
)


def test_list_files_uses_natural_order_and_joins_metadata(cataloged: Database) -> None:
    """Stems sort naturally and files without CSV rows have null metadata."""
    rows = list_files(cataloged)

    assert [row["stem"] for row in rows] == ["run-1", "run-2", "run-10"]
    assert rows[0]["lot"] == "A"
    assert rows[0]["n_steps"] == 2
    assert rows[0]["n_segments"] == 3
    assert rows[2]["lot"] is None


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
