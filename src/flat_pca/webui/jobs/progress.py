"""Progress and error files exchanged between a job process and the app."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

PROGRESS_FILE = "progress.json"
ERROR_FILE = "error.txt"
LOG_FILE = "log.txt"
REPLACE_ATTEMPTS = 10
REPLACE_RETRY_DELAY_S = 0.05


@dataclass(frozen=True)
class Progress:
    """Progress of one job.

    Attributes
    ----------
    stage : str
        Name of the current stage.
    done : int
        Completed units of the stage.
    total : int
        Total units of the stage.
    """

    stage: str
    done: int
    total: int

    @property
    def percent(self) -> float:
        """Return the completed fraction as a percentage.

        Returns
        -------
        float
            ``100 * done / total``, or ``0.0`` when ``total`` is zero.
        """
        return 0.0 if self.total <= 0 else 100.0 * self.done / self.total

    def remaining_s(self, elapsed_s: float) -> float | None:
        """Return the estimated seconds until ``done`` reaches ``total``.

        The rate of progress so far is assumed to continue:
        ``elapsed_s * (total - done) / done``.

        Parameters
        ----------
        elapsed_s : float
            Seconds spent on the ``done`` units.

        Returns
        -------
        float | None
            The estimate, or ``None`` while no unit is done.
        """
        if self.done <= 0:
            return None
        return elapsed_s * max(self.total - self.done, 0) / self.done


def write_progress(run_dir: Path, stage: str, done: int, total: int) -> None:
    """Atomically write ``progress.json``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    stage : str
        Name of the current stage.
    done : int
        Completed units of the stage.
    total : int
        Total units of the stage.
    """
    temporary = run_dir / f"{PROGRESS_FILE}.tmp"
    temporary.write_text(
        json.dumps({"stage": stage, "done": done, "total": total}), encoding="utf-8"
    )
    replace_progress_file(temporary, run_dir / PROGRESS_FILE)


def replace_progress_file(
    source: Path, target: Path, attempts: int = REPLACE_ATTEMPTS
) -> bool:
    """Move ``source`` onto ``target``, retrying while the target is in use.

    On Windows the replacement fails with ``PermissionError`` while the app is
    reading ``target``. Progress is informational, so when every attempt fails
    this update is dropped and the next one takes its place.

    Parameters
    ----------
    source : Path
        Freshly written temporary file.
    target : Path
        File to replace.
    attempts : int
        Maximum number of replacement attempts.

    Returns
    -------
    bool
        ``True`` if ``target`` was replaced, ``False`` if the update was dropped.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return True
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(REPLACE_RETRY_DELAY_S)
    return False


def read_progress(run_dir: Path) -> Progress | None:
    """Read ``progress.json``.

    Parameters
    ----------
    run_dir : Path
        Run directory.

    Returns
    -------
    Progress | None
        The latest progress, or ``None`` if the file is missing or unreadable.
    """
    try:
        raw = json.loads((run_dir / PROGRESS_FILE).read_text(encoding="utf-8"))
        return Progress(
            stage=str(raw["stage"]), done=int(raw["done"]), total=int(raw["total"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_error(run_dir: Path, message: str) -> None:
    """Write the error message of a failed job.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    message : str
        Error message shown to the user.
    """
    (run_dir / ERROR_FILE).write_text(message, encoding="utf-8")


def read_error(run_dir: Path) -> str | None:
    """Read the error message written by a failed job.

    Parameters
    ----------
    run_dir : Path
        Run directory.

    Returns
    -------
    str | None
        The message, or ``None`` if the job wrote none.
    """
    try:
        return (run_dir / ERROR_FILE).read_text(encoding="utf-8")
    except OSError:
        return None
