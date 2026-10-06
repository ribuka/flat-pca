"""Tests for the job executor with real spawned child processes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from executor_jobs import OUTPUT_FILE, crash, raise_error, write_after_delay

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.executor import JobExecutor, JobSpec
from flat_pca.webui.services.runs import get_run

Wait = Callable[..., dict[str, object]]


def _check_output(config: dict[str, object], run_dir: Path) -> dict[str, object]:
    """Finalize a ``write_after_delay`` job by validating its output."""
    text = (run_dir / OUTPUT_FILE).read_text(encoding="utf-8")
    if text != config["text"]:
        raise ValueError("unexpected output")
    return {"n_files": 1}


@pytest.fixture
def executor(database: Database, tmp_path: Path) -> Iterator[JobExecutor]:
    """Start an executor with the test job kinds."""
    started = JobExecutor(
        database,
        tmp_path / "runs",
        {
            "write": JobSpec(target=write_after_delay, finalize=_check_output),
            "crash": JobSpec(target=crash),
            "raise": JobSpec(target=raise_error),
        },
    )
    yield started
    started.shutdown()


def _write(executor: JobExecutor, seconds: float, text: str = "ok") -> str:
    """Submit a ``write_after_delay`` job."""
    return executor.submit("write", {"seconds": seconds, "text": text})


def test_job_succeeds_and_is_finalized(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """A successful child is finalized in the app process."""
    run_id = _write(executor, 0)

    run = wait_for(database, run_id)

    assert run["status"] == "succeeded"
    assert run["n_files"] == 1
    assert run["duration_s"] is not None
    assert (Path(str(run["artifact_dir"])) / "config.json").is_file()


def test_cancel_running_job_keeps_queue_and_later_submissions_working(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """Cancelling a running job does not affect queued or new jobs."""
    running = _write(executor, 60)
    queued = _write(executor, 0, "queued")
    wait_for(database, running, ("running",))
    run_dir = Path(str(get_run(database, running)["artifact_dir"]))

    assert executor.cancel(running)

    assert wait_for(database, running)["status"] == "cancelled"
    assert not run_dir.exists()
    assert wait_for(database, queued)["status"] == "succeeded"
    later = _write(executor, 0, "later")
    assert wait_for(database, later)["status"] == "succeeded"


def test_cancel_queued_job_keeps_running_job(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """Cancelling a queued job never starts it and leaves the running job alone."""
    running = _write(executor, 1)
    queued = _write(executor, 0)

    assert executor.cancel(queued)
    assert get_run(database, queued)["status"] == "cancelled"
    assert not executor.is_active(queued)
    assert wait_for(database, running)["status"] == "succeeded"
    assert get_run(database, queued)["status"] == "cancelled"


def test_crashed_child_fails_run_and_next_job_runs(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """An abnormal exit fails only that run."""
    crashed = executor.submit("crash", {})
    following = _write(executor, 0)

    run = wait_for(database, crashed)

    assert run["status"] == "failed"
    assert "code 3" in str(run["error"])
    assert wait_for(database, following)["status"] == "succeeded"


def test_raised_error_is_recorded(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """An exception in the job becomes the run's error message."""
    run = wait_for(database, executor.submit("raise", {}))

    assert run["status"] == "failed"
    assert run["error"] == "ValueError: bad input"
    assert "bad input" in (Path(str(run["artifact_dir"])) / "log.txt").read_text(
        encoding="utf-8"
    )


def test_failed_finalize_fails_run(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """A result that fails validation in the app process fails the run."""
    run_id = executor.submit("write", {"seconds": 0, "text": 1})

    run = wait_for(database, run_id)

    assert run["status"] == "failed"
    assert run["error"] == "ValueError: unexpected output"


def test_cancel_unknown_or_finished_run_returns_false(
    executor: JobExecutor, database: Database, wait_for: Wait
) -> None:
    """Only queued or running runs can be cancelled."""
    run_id = _write(executor, 0)
    wait_for(database, run_id)

    assert not executor.cancel(run_id)
    assert not executor.cancel("missing")


def test_submit_rejects_unknown_kind(executor: JobExecutor) -> None:
    """Only registered job kinds can be submitted."""
    with pytest.raises(ValueError, match="unknown job kind"):
        executor.submit("other", {})


def test_shutdown_cancels_active_jobs(
    database: Database, tmp_path: Path, wait_for: Wait
) -> None:
    """Shutting down cancels the running and queued jobs."""
    executor = JobExecutor(
        database, tmp_path / "runs", {"write": JobSpec(target=write_after_delay)}
    )
    running = _write(executor, 60)
    queued = _write(executor, 0)
    wait_for(database, running, ("running",))

    executor.shutdown()

    assert get_run(database, running)["status"] == "cancelled"
    assert get_run(database, queued)["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="shut down"):
        _write(executor, 0)
