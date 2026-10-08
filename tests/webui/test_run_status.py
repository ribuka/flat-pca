"""Tests for the elapsed and remaining time of active runs."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.progress import Progress, write_progress
from flat_pca.webui.services.runs import (
    get_run,
    insert_run,
    run_elapsed_s,
    run_status,
    update_run,
)


@pytest.mark.parametrize(
    ("done", "total", "expected"),
    [(0, 4, None), (1, 4, 30.0), (3, 4, 10.0 / 3), (4, 4, 0.0), (5, 4, 0.0)],
)
def test_remaining_time_continues_the_rate_so_far(
    done: int, total: int, expected: float | None
) -> None:
    """The estimate is ``elapsed * (total - done) / done``, unknown at zero."""
    remaining = Progress(stage="fit", done=done, total=total).remaining_s(10.0)

    if expected is None:
        assert remaining is None
    else:
        assert remaining == pytest.approx(expected)


def test_elapsed_time_counts_from_creation() -> None:
    """The elapsed time is measured from ``created_at`` and never negative."""
    created_at = datetime.fromisoformat("2026-01-02T03:04:05")
    run = {"created_at": created_at}

    assert run_elapsed_s(run, created_at + timedelta(seconds=12.5)) == 12.5
    assert run_elapsed_s(run, created_at - timedelta(seconds=1)) == 0.0


def _run(database: Database, run_dir: Path, status: str) -> dict[str, object]:
    """Insert a fit run in ``status`` and return it."""
    run_dir.mkdir()
    insert_run(database, "fit-1", "fit", {}, run_dir)
    update_run(database, "fit-1", status=status)
    run = get_run(database, "fit-1")
    assert run is not None
    return run


def test_active_run_has_elapsed_and_remaining_time(
    database: Database, tmp_path: Path
) -> None:
    """An active run's status carries its progress and both times."""
    run_dir = tmp_path / "fit-1"
    run = _run(database, run_dir, "running")
    write_progress(run_dir, "fit", 1, 4)

    status = run_status(run)

    assert status["active"] is True
    assert status["progress"] == Progress(stage="fit", done=1, total=4)
    elapsed_s = status["elapsed_s"]
    assert isinstance(elapsed_s, float) and elapsed_s >= 0.0
    assert status["remaining_s"] == pytest.approx(3 * elapsed_s)


def test_active_run_without_progress_has_no_remaining_time(
    database: Database, tmp_path: Path
) -> None:
    """A queued run without ``progress.json`` has no estimate yet."""
    status = run_status(_run(database, tmp_path / "fit-1", "queued"))

    assert status["active"] is True
    assert status["progress"] is None
    assert status["remaining_s"] is None


def test_finished_run_has_no_times(database: Database, tmp_path: Path) -> None:
    """A finished run reports neither progress nor times."""
    status = run_status(_run(database, tmp_path / "fit-1", "succeeded"))

    assert status["active"] is False
    assert status["elapsed_s"] is None
    assert status["remaining_s"] is None
