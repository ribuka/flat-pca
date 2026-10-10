"""Argument-free application factory for ``uv run -m flat_pca.webui --reload``.

uvicorn's auto-reload rebuilds the application in a worker process from an
import string, so the settings cannot be passed as an object. The launcher
stores the settings file path in ``FLAT_PCA_WEBUI_SETTINGS`` of its own
process, and the worker inherits it.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .app import create_app
from .fit_defaults import load_fit_defaults
from .settings import load_settings

SETTINGS_ENV = "FLAT_PCA_WEBUI_SETTINGS"
FACTORY_IMPORT_STRING = "flat_pca.webui.reload_factory:create_app_from_environment"


def create_app_from_environment() -> FastAPI:
    """Create the Web UI application from the settings named in the environment.

    The fit form defaults are read from ``fit_defaults.toml`` next to the
    settings file.

    Returns
    -------
    FastAPI
        Application created from ``FLAT_PCA_WEBUI_SETTINGS``.

    Raises
    ------
    RuntimeError
        If ``FLAT_PCA_WEBUI_SETTINGS`` is not set.
    """
    path = os.environ.get(SETTINGS_ENV)
    if not path:
        raise RuntimeError(f"{SETTINGS_ENV} is not set")
    return create_app(load_settings(Path(path)), load_fit_defaults(path))
