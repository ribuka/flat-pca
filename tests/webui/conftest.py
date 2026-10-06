"""Shared fixtures for the Web UI tests."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from flat_pca.webui.database import Database
from flat_pca.webui.services.runs import get_run
from flat_pca.webui.settings import Settings, load_settings
from flat_pca.webui.workspace import Workspace

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "webui"
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Copy the Web UI fixtures to a temporary directory.

    Returns
    -------
    Path
        Directory holding ``settings.toml``, ``meta.csv``, and ``data/``.
    """
    target = tmp_path / "project"
    shutil.copytree(FIXTURE_DIR, target)
    return target


@pytest.fixture
def settings(project_dir: Path) -> Settings:
    """Load the copied fixture settings.

    Returns
    -------
    Settings
        Settings whose paths point into ``project_dir``.
    """
    return load_settings(project_dir / "settings.toml")


@pytest.fixture
def database(settings: Settings) -> Iterator[Database]:
    """Open the fixture workspace database.

    Yields
    ------
    Database
        Database with the fixture metadata columns.
    """
    settings.workspace.dir.mkdir(parents=True, exist_ok=True)
    opened = Database(settings.database_path, settings.metadata_columns)
    yield opened
    opened.close()


@pytest.fixture
def workspace(settings: Settings) -> Iterator[Workspace]:
    """Open a workspace on the fixture settings.

    Yields
    ------
    Workspace
        Open workspace, closed after the test.
    """
    opened = Workspace(settings)
    yield opened
    opened.close()


def wait_for_status(
    database: Database,
    run_id: str,
    statuses: tuple[str, ...] = TERMINAL_STATUSES,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Poll a run until it reaches one of ``statuses``.

    Parameters
    ----------
    database : Database
        Workspace database.
    run_id : str
        Run to poll.
    statuses : tuple[str, ...], default TERMINAL_STATUSES
        Statuses that end the wait.
    timeout : float, default 60.0
        Seconds to wait before failing the test.

    Returns
    -------
    dict[str, object]
        The run's columns once its status is in ``statuses``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = get_run(database, run_id)
        if run is not None and run["status"] in statuses:
            return run
        time.sleep(0.05)
    pytest.fail(f"run {run_id} did not reach {statuses} within {timeout} s")


@pytest.fixture
def wait_for() -> Callable[..., dict[str, object]]:
    """Expose ``wait_for_status`` to tests.

    Returns
    -------
    Callable[..., dict[str, object]]
        ``wait_for_status``.
    """
    return wait_for_status
