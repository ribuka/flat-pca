"""Command-line entry point: ``uv run -m flat_pca.webui [--settings <path>] [--reload]``."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .app import create_app
from .fit_defaults import load_fit_defaults
from .reload_factory import FACTORY_IMPORT_STRING, SETTINGS_ENV
from .settings import SettingsError, load_settings
from .settings_files import DEFAULT_SETTINGS_PATH

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    """Validate the settings and the fit form defaults and serve the Web UI.

    With ``--reload``, the application is rebuilt by
    :func:`~flat_pca.webui.reload_factory.create_app_from_environment`
    whenever a ``*.py`` file under the ``flat_pca`` package changes.

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(prog="flat_pca.webui", description=__doc__)
    parser.add_argument(
        "--settings",
        default=DEFAULT_SETTINGS_PATH,
        help=f"path to settings.toml (default: {DEFAULT_SETTINGS_PATH.as_posix()})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart on Python code changes (development only; interrupts running jobs)",
    )
    arguments = parser.parse_args(argv)
    try:
        settings = load_settings(arguments.settings)
        fit_defaults = load_fit_defaults(arguments.settings)
    except SettingsError as error:
        parser.exit(2, f"error: {error}\n")
    if not arguments.reload:
        uvicorn.run(create_app(settings, fit_defaults), host=arguments.host, port=arguments.port)
        return
    os.environ[SETTINGS_ENV] = str(Path(arguments.settings).resolve())
    uvicorn.run(
        FACTORY_IMPORT_STRING,
        factory=True,
        reload=True,
        reload_dirs=[str(PACKAGE_DIR)],
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
