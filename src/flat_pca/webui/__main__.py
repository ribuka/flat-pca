"""Command-line entry point: ``uv run -m flat_pca.webui --settings <path>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from .app import create_app
from .settings import SettingsError, load_settings


def main(argv: Sequence[str] | None = None) -> None:
    """Validate the settings and serve the Web UI.

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(prog="flat_pca.webui", description=__doc__)
    parser.add_argument("--settings", required=True, help="path to settings.toml")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    arguments = parser.parse_args(argv)
    try:
        settings = load_settings(arguments.settings)
    except SettingsError as error:
        parser.exit(2, f"error: {error}\n")
    uvicorn.run(create_app(settings), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
