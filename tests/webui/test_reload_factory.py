"""Tests for the argument-free application factory used by ``--reload``."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from fastapi import FastAPI

from flat_pca.webui.reload_factory import (
    FACTORY_IMPORT_STRING,
    SETTINGS_ENV,
    create_app_from_environment,
)


def test_factory_import_string_names_the_factory() -> None:
    """The import string passed to uvicorn resolves to the factory."""
    module_name, attribute = FACTORY_IMPORT_STRING.split(":")

    assert getattr(import_module(module_name), attribute) is create_app_from_environment


def test_factory_reads_settings_from_environment(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory builds the app from the settings file in the environment."""
    monkeypatch.setenv(SETTINGS_ENV, str(project_dir / "settings.toml"))

    app = create_app_from_environment()

    assert isinstance(app, FastAPI)


def test_factory_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the environment variable, the factory refuses to guess a path."""
    monkeypatch.delenv(SETTINGS_ENV, raising=False)

    with pytest.raises(RuntimeError, match=SETTINGS_ENV):
        create_app_from_environment()
