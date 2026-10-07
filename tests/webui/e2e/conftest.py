"""Fixtures serving the Web UI to a browser driven by Playwright."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import uvicorn
from fit_runs import register_fit_run
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
def fitted_server_url(
    cataloged_settings: Settings, spectra_paths: list[Path]
) -> Iterator[str]:
    """Serve the Web UI on a workspace holding a succeeded fit run.

    The run (``fit-1``) imputes with the median, so the synthetic file
    lacking a row is reconstructed from imputed values.

    Yields
    ------
    str
        Base URL of the running server.
    """
    workspace = Workspace(cataloged_settings)
    try:
        register_fit_run(
            workspace.database, cataloged_settings, spectra_paths, "fit-1", "median"
        )
    finally:
        workspace.close()
    yield from serve(cataloged_settings)


@pytest.fixture(autouse=True)
def no_console_errors(page: Page) -> Iterator[None]:
    """Fail the test when the page logs a console error or throws.

    Yields
    ------
    None
        Control to the test; the collected errors are checked afterwards.
    """
    errors: list[str] = []

    def on_console(message: ConsoleMessage) -> None:
        """Record console messages of type ``error``."""
        if message.type == "error":
            errors.append(f"console: {message.text} ({message.location['url']})")

    def on_page_error(error: Error) -> None:
        """Record uncaught exceptions."""
        errors.append(f"pageerror: {error.message}")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    yield
    assert errors == []
