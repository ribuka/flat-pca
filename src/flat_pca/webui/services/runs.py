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
        ``run`` (the newest run's columns, or ``None``), ``active`` (whether
        it is queued or running), and ``progress`` (its ``progress.json``
        while active, otherwise ``None``).
    """
    run = latest_run(database, kind)
    active = run is not None and run["status"] in ACTIVE_STATUSES
    progress = read_progress(Path(str(run["artifact_dir"]))) if active else None
    return {"run": run, "active": active, "progress": progress}
