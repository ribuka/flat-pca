"""Tests for reading the catalog's file table once per change of the database."""

from __future__ import annotations

import polars as pl
import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.datatable import parse_state
from flat_pca.webui.services.file_table import file_frame, file_table_config
from flat_pca.webui.services.file_table_cache import (
    catalog_snapshot,
    file_table_context,
    snapshot_summary,
)
from flat_pca.webui.settings import Settings


def test_database_counts_writes_but_not_reads(database: Database) -> None:
    """Writes move the generation on; queries leave it."""
    start = database.generation

    database.fetch_dicts("SELECT count(*) AS n FROM files")
    database.fetch_frame("SELECT stem FROM files")
    assert database.generation == start
    database.execute("DELETE FROM file_metadata")
    assert database.generation == start + 1
    with database.transaction() as connection:
        connection.execute("DELETE FROM segments")
    assert database.generation == start + 2


def test_fetch_frame_keeps_the_column_types(cataloged: Database) -> None:
    """The rows come as a polars frame with DuckDB's types."""
    frame = cataloged.fetch_frame("SELECT stem, n_rows, wavelength_min FROM files ORDER BY stem")

    assert frame.schema == pl.Schema(
        {"stem": pl.String, "n_rows": pl.Int64, "wavelength_min": pl.Float64}
    )
    assert frame["stem"].to_list() == ["run-1", "run-10", "run-2"]


def test_snapshot_is_kept_until_the_database_changes(cataloged: Database) -> None:
    """The rows are read again only after a write."""
    first = catalog_snapshot(cataloged)

    assert catalog_snapshot(cataloged) is first
    assert first.frame.equals(file_frame(cataloged))
    assert first.options == {"lot": ["A", "B"]}
    cataloged.execute("DELETE FROM files WHERE stem = 'run-2'")
    second = catalog_snapshot(cataloged)
    assert second is not first
    assert second.frame["stem"].to_list() == ["run-1", "run-10"]


def test_summary_is_made_once_per_snapshot_and_settings(
    cataloged: Database, settings: Settings
) -> None:
    """The column menus and histograms are counted once per catalog."""
    config = file_table_config(settings.metadata_columns)
    snapshot = catalog_snapshot(cataloged)

    summary = snapshot_summary(snapshot, config)

    assert snapshot_summary(snapshot, config) is summary
    assert summary.options == {"lot": ["A", "B"]}
    assert summary.counts.nulls["lot"] == 1
    assert "n_rows" in summary.histograms


@pytest.mark.parametrize("with_warnings", [True, False])
def test_file_table_context_pages_the_snapshot(
    cataloged: Database, settings: Settings, with_warnings: bool
) -> None:
    """The context holds the settings, one view, and the warnings when asked."""
    config = file_table_config(settings.metadata_columns)

    context = file_table_context(
        cataloged, config, parse_state({"files.order": ["desc"]}, config), with_warnings
    )

    assert context["table"] is config
    assert [row["stem"] for row in context["view"].rows] == ["run-10", "run-2", "run-1"]
    assert (context["warnings"] is not None) == with_warnings
