"""Tests for the shown-file dialog's table: its settings and its rows."""

from __future__ import annotations

from flat_pca.webui.database import Database
from flat_pca.webui.services.file_table import file_table_config
from flat_pca.webui.services.view_file_table import (
    VIEW_FILE_FORM_ID,
    VIEW_FILE_TABLE_ID,
    VIEW_FILE_TABLE_URL,
    view_file_frame,
    view_file_table_config,
)
from flat_pca.webui.settings import Settings


def test_config_has_the_file_table_columns_with_its_own_id_and_limit(
    settings: Settings,
) -> None:
    """The columns are the data selection table's; the id, URL, and limit differ."""
    files = file_table_config(settings.metadata_columns)

    config = view_file_table_config(settings.metadata_columns, max_files=5)

    assert config.columns == files.columns
    assert config.default_sort == files.default_sort
    assert config.table_id == VIEW_FILE_TABLE_ID != files.table_id
    assert config.url == VIEW_FILE_TABLE_URL
    assert config.selection_form == VIEW_FILE_FORM_ID
    assert config.max_selected == 5


def test_frame_keeps_the_targets_in_order_with_blank_uncataloged_rows(
    cataloged: Database,
) -> None:
    """Only the given files are rows; a file missing from the catalog has nulls."""
    frame = view_file_frame(cataloged, ["run-2", "ghost", "run-10"])

    assert frame["stem"].to_list() == ["run-2", "ghost", "run-10"]
    assert frame.columns == [
        "stem", "path", "n_rows", "n_steps", "n_segments", "lot", "date", "yield_pct"
    ]
    ghost = frame.row(1, named=True)
    assert ghost["path"] == ""
    assert all(ghost[name] is None for name in ("n_rows", "lot", "date", "yield_pct"))
    assert frame.row(0, named=True)["path"].endswith("run-2.parquet")
    assert frame.row(0, named=True)["n_rows"] is not None
