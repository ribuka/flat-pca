"""FIFO job executor running each job in its own spawned child process."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing.process import BaseProcess
from pathlib import Path

from loguru import logger

from ..database import Database
from ..services.runs import insert_run, update_run
from .progress import LOG_FILE, read_error, write_error

JobTarget = Callable[[dict[str, object], Path], None]
JobFinalizer = Callable[[dict[str, object], Path], Mapping[str, object]]
CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class JobSpec:
    """How to run and register one kind of job.

    Attributes
    ----------
    target : Callable[[dict[str, object], Path], None]
        Module-level function run in the child process as
        ``target(config, run_dir)``. It writes its results to ``run_dir``.
    finalize : Callable[[dict[str, object], Path], Mapping[str, object]] | None
        Called in the app process after the child exits with code 0, as
        ``finalize(config, run_dir)``. It validates the results and registers
        them in the database, and returns extra ``runs`` columns to store.
        Raising marks the run as failed.
    """

    target: JobTarget
    finalize: JobFinalizer | None = None


@dataclass
class _Job:
    """One submitted job and its execution state."""

    run_id: str
    spec: JobSpec
    config: dict[str, object]
    run_dir: Path
    process: BaseProcess | None = None
    cancelled: bool = False
    started_at: float = field(default=0.0)


def _run_in_child(target: JobTarget, config: dict[str, object], run_dir: Path) -> None:
    """Run a job target in the child process and record a failure message.

    Parameters
    ----------
    target : Callable[[dict[str, object], Path], None]
        Job function.
    config : dict[str, object]
        Job configuration.
    run_dir : Path
        Run directory.
    """
    logger.remove()
    logger.add(run_dir / LOG_FILE, level="INFO")
    try:
        target(config, run_dir)
    except BaseException as error:  # noqa: BLE001 - report every failure to the app
        logger.error(traceback.format_exc())
        write_error(run_dir, f"{type(error).__name__}: {error}")
        sys.exit(1)


def new_run_id() -> str:
    """Return a new sortable, unique run identifier.

    Returns
    -------
    str
        ``YYYYmmdd-HHMMSS-`` followed by eight random hexadecimal digits.
    """
    return f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


class JobExecutor:
    """Run jobs one at a time, in submission order, in spawned processes.

    A single monitor thread starts the next queued job only after the
    previous child process has exited (``join``), so at most one job runs at
    a time and each child's memory is fully released. A crashing or
    terminated child affects only its own run.

    Parameters
    ----------
    database : Database
        Workspace database receiving run states.
    runs_dir : Path
        Directory under which each run gets ``{runs_dir}/{run_id}``.
    specs : Mapping[str, JobSpec]
        Job kinds that can be submitted.
    """

    def __init__(
        self,
        database: Database,
        runs_dir: Path,
        specs: Mapping[str, JobSpec],
    ) -> None:
        self._database = database
        self._runs_dir = runs_dir
        self._specs = dict(specs)
        self._context = multiprocessing.get_context("spawn")
        self._queue: deque[_Job] = deque()
        self._current: _Job | None = None
        self._condition = threading.Condition()
        self._stopping = False
        self._thread = threading.Thread(
            target=self._monitor, name="flat-pca-job-monitor", daemon=True
        )
        self._thread.start()

    def submit(self, kind: str, config: Mapping[str, object]) -> str:
        """Queue a job.

        Parameters
        ----------
        kind : str
            Registered job kind.
        config : Mapping[str, object]
            JSON-serializable job configuration.

        Returns
        -------
        str
            The new run's identifier.

        Raises
        ------
        ValueError
            If ``kind`` is not registered.
        RuntimeError
            If the executor has been shut down.
        """
        if kind not in self._specs:
            raise ValueError(f"unknown job kind: {kind!r}")
        run_id = new_run_id()
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True)
        job_config = dict(config)
        (run_dir / CONFIG_FILE).write_text(
            json.dumps(job_config, indent=2), encoding="utf-8"
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("job executor is shut down")
            insert_run(self._database, run_id, kind, job_config, run_dir)
            self._queue.append(_Job(run_id, self._specs[kind], job_config, run_dir))
            self._condition.notify_all()
        return run_id

    def cancel(self, run_id: str) -> bool:
        """Cancel a queued or running job.

        A queued job is removed from the queue and never started. A running
        job's child process is terminated; the monitor thread then waits for
        it to exit, deletes its partial results, and marks the run
        ``cancelled`` before starting the next job.

        Parameters
        ----------
        run_id : str
            Run to cancel.

        Returns
        -------
        bool
            ``True`` if the run was queued or running, ``False`` otherwise.
        """
        with self._condition:
            for job in self._queue:
                if job.run_id == run_id:
                    self._queue.remove(job)
                    update_run(self._database, run_id, status="cancelled")
                    shutil.rmtree(job.run_dir, ignore_errors=True)
                    return True
            job = self._current
            if job is None or job.run_id != run_id:
                return False
            job.cancelled = True
            if job.process is not None:
                job.process.terminate()
            return True

    def is_active(self, run_id: str) -> bool:
        """Return whether a run is queued or running in this executor.

        Parameters
        ----------
        run_id : str
            Run to check.

        Returns
        -------
        bool
            ``True`` while the run is queued or running.
        """
        with self._condition:
            current = self._current is not None and self._current.run_id == run_id
            return current or any(job.run_id == run_id for job in self._queue)

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel every queued and running job and stop the monitor thread.

        Parameters
        ----------
        timeout : float, default 10.0
            Seconds to wait for the monitor thread to exit.
        """
        with self._condition:
            self._stopping = True
            queued = list(self._queue)
            self._queue.clear()
            if self._current is not None:
                self._current.cancelled = True
                if self._current.process is not None:
                    self._current.process.terminate()
            self._condition.notify_all()
        for job in queued:
            update_run(self._database, job.run_id, status="cancelled")
            shutil.rmtree(job.run_dir, ignore_errors=True)
        self._thread.join(timeout)

    def _next_job(self) -> _Job | None:
        """Wait for a queued job and start its child process.

        Returns
        -------
        _Job | None
            The started job, or ``None`` when the executor is stopping.
        """
        with self._condition:
            while True:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return None
                job = self._queue.popleft()
                update_run(self._database, job.run_id, status="running")
                job.started_at = time.monotonic()
                process = self._context.Process(
                    target=_run_in_child,
                    args=(job.spec.target, job.config, job.run_dir),
                    name=f"flat-pca-job-{job.run_id}",
                )
                try:
                    process.start()
                except Exception as error:  # noqa: BLE001 - fail only this run
                    logger.exception(f"starting run {job.run_id} failed")
                    update_run(
                        self._database,
                        job.run_id,
                        status="failed",
                        error=f"cannot start job process: {error}",
                    )
                    continue
                job.process = process
                self._current = job
                return job

    def _monitor(self) -> None:
        """Start queued jobs one by one and record each outcome."""
        while (job := self._next_job()) is not None:
            assert job.process is not None
            job.process.join()
            with self._condition:
                self._finish(job)
                self._current = None
                job.process.close()

    def _finish(self, job: _Job) -> None:
        """Record the outcome of a job whose child process has exited.

        Parameters
        ----------
        job : _Job
            Finished job.
        """
        assert job.process is not None
        duration = time.monotonic() - job.started_at
        if job.cancelled:
            shutil.rmtree(job.run_dir, ignore_errors=True)
            update_run(
                self._database, job.run_id, status="cancelled", duration_s=duration
            )
            return
        exitcode = job.process.exitcode
        if exitcode != 0:
            error = (
                read_error(job.run_dir) or f"job process exited with code {exitcode}"
            )
            update_run(
                self._database,
                job.run_id,
                status="failed",
                error=error,
                duration_s=duration,
            )
            return
        try:
            fields = (
                {}
                if job.spec.finalize is None
                else job.spec.finalize(job.config, job.run_dir)
            )
        except Exception as error:  # noqa: BLE001 - any validation failure fails the run
            logger.exception(f"finalizing run {job.run_id} failed")
            update_run(
                self._database,
                job.run_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                duration_s=duration,
            )
            return
        update_run(
            self._database,
            job.run_id,
            status="succeeded",
            duration_s=duration,
            **fields,
        )
