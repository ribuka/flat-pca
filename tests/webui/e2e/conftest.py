"""Fixtures serving the Web UI to a browser driven by Playwright."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import executor_jobs
import pytest
import uvicorn
from fit_runs import register_fit_run, register_transform_run
from playwright.sync_api import ConsoleMessage, Error, Page

from flat_pca.webui.app import create_app
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import Workspace

STARTUP_TIMEOUT_S = 30.0
SHUTDOWN_TIMEOUT_S = 30.0


def serve(settings: Settings) -> Iterator[str]:
    """Serve the Web UI with uvicorn in a background thread.

    The listening socket is bound before the server starts, so the port is
    reserved without a race.

    Parameters
    ----------
    settings : Settings
        Settings of the served application.

    Yields
    ------
    str
        Base URL of the running server; the server stops on exit.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), log_level="warning")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                pytest.fail("the Web UI server did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(SHUTDOWN_TIMEOUT_S)
        sock.close()


@pytest.fixture
def server_url(settings: Settings) -> Iterator[str]:
    """Serve the Web UI on the fixture settings without a catalog.

    Yields
    ------
    str
        Base URL of the running server.
    """
    yield from serve(settings)


@pytest.fixture
def cataloged_settings(
    settings: Settings, wait_for: Callable[..., dict[str, object]]
) -> Settings:
    """Run one successful catalog update on the fixture workspace.

    Returns
    -------
    Settings
        The fixture settings, whose workspace now holds a catalog.
    """
    workspace = Workspace(settings)
    try:
        run = wait_for(workspace.database, workspace.submit_catalog())
    finally:
        workspace.close()
    assert run["status"] == "succeeded"
    return settings


@pytest.fixture
def cataloged_server_url(cataloged_settings: Settings) -> Iterator[str]:
    """Serve the Web UI on a workspace that already holds a catalog.

    Yields
    ------
    str
        Base URL of the running server.
    """
    yield from serve(cataloged_settings)


@pytest.fixture
def slow_fit_server_url(
    cataloged_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """Serve the Web UI with a fit job that goes slowly through its stages and fails.

    The job is ``executor_jobs.fit_in_slow_stages``, so the browser can watch
    an active fit run.

    Yields
    ------
    str
        Base URL of the running server.
    """
    monkeypatch.setattr(
        "flat_pca.webui.workspace.run_fit", executor_jobs.fit_in_slow_stages
    )
    yield from serve(cataloged_settings)

def register_fit_runs(
    settings: Settings,
    spectra_paths: list[Path],
    run_ids: tuple[str, ...],
    transform: bool = True,
) -> None:
    """Register succeeded fit runs, oldest first, that impute with the median.

    Parameters
    ----------
    settings : Settings
        Settings of the workspace.
    spectra_paths : list[Path]
        Synthetic spectra files.
    run_ids : tuple[str, ...]
        Identifiers of the fit runs (``fit-N``) in creation order.
    transform : bool, default True
        Whether each fit run is followed by a succeeded transform run of its
        own files (``tr-N``), which the display screens show.
    """
    workspace = Workspace(settings)
    try:
        for run_id in run_ids:
            register_fit_run(workspace.database, settings, spectra_paths, run_id, "median")
            if transform:
                register_transform_run(
                    workspace.database,
                    settings,
                    run_id,
                    spectra_paths,
                    run_id.replace("fit-", "tr-"),
                )
    finally:
        workspace.close()


@pytest.fixture
def fitted_server_url(
    cataloged_settings: Settings, spectra_paths: list[Path]
) -> Iterator[str]:
    """Serve the Web UI on a workspace holding a succeeded fit run and its transform run.

    The fit run (``fit-1``) imputes with the median, so the synthetic file
    lacking a row is reconstructed from imputed values. The transform run
    (``tr-1``) of the same files is shown.

    Yields
    ------
    str
        Base URL of the running server.
    """
    register_fit_runs(cataloged_settings, spectra_paths, ("fit-1",))
    yield from serve(cataloged_settings)


@pytest.fixture
def slow_transform_server_url(
    cataloged_settings: Settings,
    spectra_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    """Serve the Web UI with the fit run ``fit-1`` and a slow, failing transform job.

    The transform job is ``executor_jobs.fit_in_slow_stages``, so the browser
    can watch an active transform run.

    Yields
    ------
    str
        Base URL of the running server.
    """
    register_fit_runs(cataloged_settings, spectra_paths, ("fit-1",), transform=False)
    monkeypatch.setattr(
        "flat_pca.webui.workspace.run_transform", executor_jobs.fit_in_slow_stages
    )
    yield from serve(cataloged_settings)


@pytest.fixture
def two_fits_server_url(settings: Settings, spectra_paths: list[Path]) -> Iterator[str]:
    """Serve the Web UI with the fit runs ``fit-1`` and ``fit-2`` and their ``tr-1`` and ``tr-2``.

    Yields
    ------
    str
        Base URL of the running server.
    """
    register_fit_runs(settings, spectra_paths, ("fit-1", "fit-2"))
    yield from serve(settings)


@pytest.fixture
def one_file_server_url(settings: Settings, spectra_paths: list[Path]) -> Iterator[str]:
    """Serve the Web UI showing one file at a time, with ``fit-1`` and its ``tr-1``.

    Yields
    ------
    str
        Base URL of the running server.
    """
    limited = settings.model_copy(
        update={"ui": settings.ui.model_copy(update={"explore_max_files": 1})}
    )
    register_fit_runs(limited, spectra_paths, ("fit-1",))
    yield from serve(limited)


@pytest.fixture
def expected_console_errors() -> list[str]:
    """Return texts of console errors that a test expects to be logged.

    A test that makes a request fail on purpose appends a text contained in
    the error that the browser logs for it.

    Returns
    -------
    list[str]
        Initially empty.
    """
    return []


@pytest.fixture(autouse=True)
def no_console_errors(page: Page, expected_console_errors: list[str]) -> Iterator[None]:
    """Fail the test when the page logs an unexpected console error or throws.

    Console errors containing a text of ``expected_console_errors`` are
    allowed.

    Yields
    ------
    None
        Control to the test; the collected errors are checked afterwards.
    """
    errors: list[str] = []

    def on_console(message: ConsoleMessage) -> None:
        """Record console messages of type ``error``."""
        if message.type == "error" and not any(
            text in message.text for text in expected_console_errors
        ):
            errors.append(f"console: {message.text} ({message.location['url']})")

    def on_page_error(error: Error) -> None:
        """Record uncaught exceptions."""
        errors.append(f"pageerror: {error.message}")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    yield
    assert errors == []
