"""Reading and writing rows of the ``runs`` table."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..database import Database
from ..jobs.progress import read_progress

RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ACTIVE_STATUSES: tuple[RunStatus, ...] = ("queued", "running")
INTERRUPTED_ERROR = "interrupted: the application stopped while the run was active"


def insert_run(
    database: Database,
    run_id: str,
    kind: str,
    config: Mapping[str, object],
    artifact_dir: Path,
) -> None:
    """Insert a new run in the ``queued`` state.

    Parameters
    ----------
    database : Database
        Workspace database.
    run_id : str
        Unique run identifier.
    kind : str
        Job kind, such as ``"catalog"``.
    config : Mapping[str, object]
        JSON-serializable job configuration.
    artifact_dir : Path
        Run directory receiving the job's files.
    """
    created_at = datetime.now().astimezone().replace(tzinfo=None)
    database.execute(
        "INSERT INTO runs (run_id, kind, created_at, status, config_json, artifact_dir) "
        "VALUES (?, ?, ?, 'queued', ?, ?)",
        [run_id, kind, created_at, json.dumps(config), str(artifact_dir)],
    )


def update_run(database: Database, run_id: str, **fields: object) -> None:
    """Update columns of one run.

    Parameters
    ----------
    database : Database
        Workspace database.
    run_id : str
        Run to update.
    **fields : object
        Column values keyed by ``runs`` column name, such as ``status``,
        ``error``, ``duration_s``, ``n_files``, ``n_features``, or
        ``n_components``.

    Raises
    ------
    ValueError
        If no field or an unknown column is given.
    """
    allowed = {"status", "error", "duration_s", "n_files", "n_features", "n_components"}
    unknown = set(fields) - allowed
    if not fields or unknown:
        raise ValueError(f"invalid run fields: {sorted(unknown) or 'none'}")
    assignments = ", ".join(f"{name} = ?" for name in fields)
    database.execute(
        f"UPDATE runs SET {assignments} WHERE run_id = ?",
        [*fields.values(), run_id],
    )


def get_run(database: Database, run_id: str) -> dict[str, object] | None:
    """Return one run.

    Parameters
    ----------
    database : Database
        Workspace database.
    run_id : str
        Run to read.

    Returns
    -------
    dict[str, object] | None
        The run's columns, or ``None`` if it does not exist.
    """
    rows = database.fetch_dicts("SELECT * FROM runs WHERE run_id = ?", [run_id])
    return rows[0] if rows else None


def latest_run(database: Database, kind: str) -> dict[str, object] | None:
    """Return the most recently created run of one kind.

    Parameters
    ----------
    database : Database
        Workspace database.
    kind : str
        Job kind.

    Returns
    -------
    dict[str, object] | None
        The newest run's columns, or ``None`` if no run of ``kind`` exists.
    """
    rows = database.fetch_dicts(
        "SELECT * FROM runs WHERE kind = ? ORDER BY created_at DESC, run_id DESC LIMIT 1",
        [kind],
    )
    return rows[0] if rows else None


def fail_interrupted_runs(database: Database) -> int:
    """Mark runs left ``queued`` or ``running`` by a previous process as failed.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    int
        Number of runs marked as failed.
    """
    rows = database.fetch_dicts(
        "UPDATE runs SET status = 'failed', error = ? "
        "WHERE status IN ('queued', 'running') RETURNING run_id",
        [INTERRUPTED_ERROR],
    )
    return len(rows)


def list_runs(
    database: Database, kind: str, limit: int = 100, status: RunStatus | None = None
) -> list[dict[str, object]]:
    """Return the runs of one kind, newest first.

    Parameters
    ----------
    database : Database
        Workspace database.
    kind : str
        Job kind.
    limit : int, default 100
        Maximum number of runs.
    status : RunStatus | None, default None
        Only runs in this status, applied before ``limit``; ``None`` for all.

    Returns
    -------
    list[dict[str, object]]
        Each run's columns.
    """
    return database.fetch_dicts(
        "SELECT * FROM runs WHERE kind = ? AND (? IS NULL OR status = ?) "
        "ORDER BY created_at DESC, run_id DESC LIMIT ?",
        [kind, status, status, limit],
    )


def list_succeeded_runs(
    database: Database, kind: str, requested: str | None
) -> list[dict[str, object]]:
    """Return the succeeded runs of one kind to choose from.

    Parameters
    ----------
    database : Database
        Workspace database.
    kind : str
        Job kind.
    requested : str | None
        Explicitly requested run. It is added when it is a succeeded run of
        ``kind`` older than the listed ones.

    Returns
    -------
    list[dict[str, object]]
        The newest succeeded runs, then the requested run if needed.
    """
    runs = list_runs(database, kind, status="succeeded")
    if requested is not None and all(run["run_id"] != requested for run in runs):
        run = get_run(database, requested)
        if run is not None and run["kind"] == kind and run["status"] == "succeeded":
            runs.append(run)
    return runs


def latest_run_status(database: Database, kind: str) -> dict[str, object]:
    """Return the latest run of one kind with its progress.

    Parameters
    ----------
    database : Database
        Workspace database.
    kind : str
        Job kind.

    Returns
    -------
    dict[str, object]
        See ``run_status``.
    """
    return run_status(latest_run(database, kind))


def run_elapsed_s(run: Mapping[str, object], now: datetime | None = None) -> float:
    """Return the seconds since a run was created.

    Parameters
    ----------
    run : Mapping[str, object]
        The run's columns; ``created_at`` is a naive local time, as written
        by ``insert_run``.
    now : datetime | None, default None
        Current naive local time; ``None`` for the clock's time.

    Returns
    -------
    float
        Elapsed seconds, never negative.

    Raises
    ------
    TypeError
        If ``created_at`` is not a ``datetime``.
    """
    current = datetime.now().astimezone().replace(tzinfo=None) if now is None else now
    created_at = run["created_at"]
    if not isinstance(created_at, datetime):
        raise TypeError(f"created_at is not a datetime: {created_at!r}")
    return max((current - created_at).total_seconds(), 0.0)


def run_status(run: dict[str, object] | None) -> dict[str, object]:
    """Return a run with its progress.

    Parameters
    ----------
    run : dict[str, object] | None
        The run's columns, or ``None``.

    Returns
    -------
    dict[str, object]
        ``run`` (the given run), ``active`` (whether it is queued or
        running), ``progress`` (its ``progress.json`` while active,
        otherwise ``None``), ``elapsed_s`` (seconds since the run was
        created while active, otherwise ``None``), and ``remaining_s`` (the
        estimated seconds left from ``progress``, or ``None`` while active
        without done units or when inactive).
    """
    active = run is not None and run["status"] in ACTIVE_STATUSES
    if not active or run is None:
        return {
            "run": run,
            "active": False,
            "progress": None,
            "elapsed_s": None,
            "remaining_s": None,
        }
    progress = read_progress(Path(str(run["artifact_dir"])))
    elapsed_s = run_elapsed_s(run)
    return {
        "run": run,
        "active": True,
        "progress": progress,
        "elapsed_s": elapsed_s,
        "remaining_s": None if progress is None else progress.remaining_s(elapsed_s),
    }
