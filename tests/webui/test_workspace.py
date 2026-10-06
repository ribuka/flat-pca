"""Tests for the ``Workspace`` and its startup handling of runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flat_pca.webui.__main__ import main
from flat_pca.webui.database import Database
from flat_pca.webui.services.runs import (
    INTERRUPTED_ERROR,
    get_run,
    insert_run,
    update_run,
)
from flat_pca.webui.settings import MetadataColumnSettings, Settings
from flat_pca.webui.workspace import CATALOG_JOB, Workspace

Wait = Callable[..., dict[str, object]]


def test_workspace_fails_runs_left_active_by_previous_process(
    settings: Settings, tmp_path: Path
) -> None:
    """Queued and running runs from an earlier process are marked failed."""
    database = Database(settings.database_path, settings.metadata_columns)
    for run_id, status in (("a", "running"), ("b", "queued"), ("c", "succeeded")):
        insert_run(database, run_id, CATALOG_JOB, {}, tmp_path)
        update_run(database, run_id, status=status)
    database.close()

    workspace = Workspace(settings)
    try:
        statuses = {
            run_id: get_run(workspace.database, run_id) for run_id in ("a", "b", "c")
        }
    finally:
        workspace.close()

    assert statuses["a"]["status"] == "failed"
    assert statuses["a"]["error"] == INTERRUPTED_ERROR
    assert statuses["b"]["status"] == "failed"
    assert statuses["c"]["status"] == "succeeded"


def test_database_rebuilds_file_metadata_when_column_type_changes(
    settings: Settings,
) -> None:
    """Changing a column's type empties ``file_metadata`` with the new SQL type."""
    category = {"lot": MetadataColumnSettings(type="category")}
    database = Database(settings.database_path, category)
    database.execute("INSERT INTO file_metadata VALUES ('run-1', 'A')")
    database.close()

    database = Database(
        settings.database_path, {"lot": MetadataColumnSettings(type="number")}
    )
    try:
        rows = database.fetch_dicts("SELECT * FROM file_metadata")
        types = database.fetch_dicts(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'file_metadata' AND column_name = 'lot'"
        )
    finally:
        database.close()

    assert rows == []
    assert types == [{"data_type": "DOUBLE"}]


def test_database_keeps_file_metadata_when_columns_match(settings: Settings) -> None:
    """Reopening with the same columns keeps the imported rows."""
    category = {"lot": MetadataColumnSettings(type="category")}
    database = Database(settings.database_path, category)
    database.execute("INSERT INTO file_metadata VALUES ('run-1', 'A')")
    database.close()

    database = Database(settings.database_path, category)
    try:
        rows = database.fetch_dicts("SELECT * FROM file_metadata")
    finally:
        database.close()

    assert rows == [{"stem": "run-1", "lot": "A"}]


def test_workspace_rebuilds_file_metadata_when_columns_change(
    settings: Settings,
) -> None:
    """Changing the configured metadata columns recreates ``file_metadata``."""
    Database(settings.database_path, {}).close()

    database = Database(settings.database_path, settings.metadata_columns)
    try:
        columns = database.fetch_dicts(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'file_metadata' ORDER BY ordinal_position"
        )
    finally:
        database.close()

    assert [row["column_name"] for row in columns] == [
        "stem",
        "lot",
        "date",
        "yield_pct",
    ]


def test_submit_catalog_runs_job_and_reuses_active_run(
    workspace: Workspace, wait_for: Wait
) -> None:
    """A catalog run registers the files; a second request reuses an active run."""
    first = workspace.submit_catalog()
    second = workspace.submit_catalog()

    run = wait_for(workspace.database, first)

    assert run["status"] == "succeeded"
    assert run["n_files"] == 3
    assert second == first


def test_main_exits_on_invalid_settings(tmp_path: Path) -> None:
    """Invalid settings stop the server before it starts."""
    path = tmp_path / "s.toml"
    path.write_text("[workspace]\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main(["--settings", str(path)])

    assert raised.value.code == 2
