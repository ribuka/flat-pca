"""Tests for the catalog job and its registration in the database."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.catalog import (
    FILES_RESULT,
    build_catalog_config,
    run_catalog,
    scan_parquet_file,
)
from flat_pca.webui.jobs.progress import read_progress
from flat_pca.webui.services.catalog_store import apply_catalog_result, known_files
from flat_pca.webui.settings import Settings


def _catalog(settings: Settings, database: Database, run_dir: Path) -> pl.DataFrame:
    """Run one catalog job directly and register its result.

    Returns
    -------
    pl.DataFrame
        The job's ``files.parquet``.
    """
    run_dir.mkdir(parents=True)
    config = build_catalog_config(settings, known_files(database))
    run_catalog(config, run_dir)
    apply_catalog_result(database, config, run_dir)
    return pl.read_parquet(run_dir / FILES_RESULT)


def test_scan_parquet_file_summarizes_schema_and_segments(settings: Settings) -> None:
    """Wavelengths come from the schema and segments from Step/Sequence."""
    summary, segments = scan_parquet_file(settings.data.root / "run-1.parquet")

    assert summary == {
        "n_wavelengths": 3,
        "wavelength_min": 400.0,
        "wavelength_max": 402.5,
        "n_rows": 7,
    }
    assert segments.rows() == [
        ("run-1", 1, 1, 3, 1.0),
        ("run-1", 2, 1, 2, 0.5),
        ("run-1", 2, 2, 2, 0.5),
    ]


def test_catalog_registers_files_segments_and_metadata(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    """A first catalog run scans every file and imports the CSV."""
    files = _catalog(settings, database, tmp_path / "run1")

    assert files["rescanned"].to_list() == [True, True, True]
    assert database.fetch_dicts("SELECT stem, n_rows FROM files ORDER BY stem") == [
        {"stem": "run-1", "n_rows": 7},
        {"stem": "run-10", "n_rows": 7},
        {"stem": "run-2", "n_rows": 7},
    ]
    assert database.fetch_dicts("SELECT count(*) AS n FROM segments") == [{"n": 9}]
    assert database.fetch_dicts(
        "SELECT stem, lot FROM file_metadata ORDER BY stem"
    ) == [
        {"stem": "ghost", "lot": "A"},
        {"stem": "run-1", "lot": "A"},
        {"stem": "run-2", "lot": "B"},
    ]
    progress = read_progress(tmp_path / "run1")
    assert progress is not None
    assert (progress.stage, progress.done, progress.total) == ("metadata", 1, 1)


def test_catalog_rescans_only_changed_files(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    """Unchanged files keep their rows; changed and removed files are updated."""
    _catalog(settings, database, tmp_path / "run1")
    changed = settings.data.root / "run-2.parquet"
    pl.read_parquet(changed).head(3).write_parquet(changed)
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    (settings.data.root / "sub" / "run-10.parquet").unlink()

    files = _catalog(settings, database, tmp_path / "run2")

    assert dict(files.select("stem", "rescanned").iter_rows()) == {
        "run-1": False,
        "run-2": True,
    }
    assert database.fetch_dicts("SELECT stem, n_rows FROM files ORDER BY stem") == [
        {"stem": "run-1", "n_rows": 7},
        {"stem": "run-2", "n_rows": 3},
    ]
    assert database.fetch_dicts(
        "SELECT stem, count(*) AS n FROM segments GROUP BY stem ORDER BY stem"
    ) == [{"stem": "run-1", "n": 3}, {"stem": "run-2", "n": 1}]


def test_catalog_rescans_moved_file(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    """A file moved to another path under the root is scanned again."""
    _catalog(settings, database, tmp_path / "run1")
    source = settings.data.root / "sub" / "run-10.parquet"
    shutil.move(source, settings.data.root / "run-10.parquet")

    files = _catalog(settings, database, tmp_path / "run2")

    assert dict(files.select("stem", "rescanned").iter_rows())["run-10"] is True
    assert database.fetch_dicts("SELECT path FROM files WHERE stem = 'run-10'") == [
        {"path": str((settings.data.root / "run-10.parquet").resolve())}
    ]


def test_catalog_rejects_duplicate_stems(settings: Settings, tmp_path: Path) -> None:
    """Two files with the same stem under the root are an error."""
    shutil.copy(settings.data.root / "run-1.parquet", settings.data.root / "sub")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="stems must be unique"):
        run_catalog(build_catalog_config(settings, {}), run_dir)


def test_catalog_rejects_invalid_parquet_schema(
    settings: Settings, tmp_path: Path
) -> None:
    """A file without the required columns fails the job."""
    pl.DataFrame({"Time": [0.0]}).write_parquet(settings.data.root / "bad.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="required columns missing"):
        run_catalog(build_catalog_config(settings, {}), run_dir)


def test_apply_catalog_result_requires_result_files(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    """Registration fails when the job did not write its results."""
    with pytest.raises(FileNotFoundError, match=FILES_RESULT):
        apply_catalog_result(database, build_catalog_config(settings, {}), tmp_path)
