"""The ``Workspace`` holding all Web UI state."""

from __future__ import annotations

import threading
from functools import partial

from loguru import logger

from .database import Database
from .jobs.catalog import build_catalog_config, run_catalog
from .jobs.executor import JobExecutor, JobSpec
from .services.catalog_store import apply_catalog_result, known_files
from .services.runs import ACTIVE_STATUSES, fail_interrupted_runs, latest_run
from .services.selection import FileSelection
from .settings import Settings

CATALOG_JOB = "catalog"


class Workspace:
    """Database connection, display cache, file selection, and job executor.

    All Web UI state lives in one ``Workspace`` instead of module globals.
    Opening a workspace marks runs left ``queued`` or ``running`` by a
    previous process as failed.

    Parameters
    ----------
    settings : Settings
        Validated application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self.database = Database(settings.database_path, settings.metadata_columns)
        interrupted = fail_interrupted_runs(self.database)
        if interrupted:
            logger.warning(f"marked {interrupted} interrupted runs as failed")
        self.cache: dict[str, object] = {}
        self.selection = FileSelection()
        self._submit_lock = threading.Lock()
        self.executor = JobExecutor(
            self.database,
            settings.runs_dir,
            {
                CATALOG_JOB: JobSpec(
                    target=run_catalog,
                    finalize=partial(apply_catalog_result, self.database),
                )
            },
        )

    def submit_catalog(self) -> str:
        """Queue a catalog update unless one is already queued or running.

        Returns
        -------
        str
            The active or newly queued catalog run's identifier.
        """
        with self._submit_lock:
            current = latest_run(self.database, CATALOG_JOB)
            if current is not None and current["status"] in ACTIVE_STATUSES:
                return str(current["run_id"])
            config = build_catalog_config(self.settings, known_files(self.database))
            return self.executor.submit(CATALOG_JOB, config)

    def close(self) -> None:
        """Cancel active jobs and close the database."""
        self.executor.shutdown()
        self.database.close()
