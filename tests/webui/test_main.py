"""Tests for the ``uv run -m flat_pca.webui`` command-line entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from flat_pca.webui import __main__ as entry
from flat_pca.webui.reload_factory import FACTORY_IMPORT_STRING, SETTINGS_ENV

Call = tuple[tuple[object, ...], dict[str, object]]


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Record ``uvicorn.run`` calls instead of serving.

    The settings environment variable is cleared and restored around the test.

    Returns
    -------
    list[Call]
        Positional and keyword arguments of each call.
    """
    calls: list[Call] = []
    monkeypatch.setattr(
        entry.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setenv(SETTINGS_ENV, "")  # registers the original value for restoring
    monkeypatch.delenv(SETTINGS_ENV)
    return calls


@pytest.mark.parametrize("reload", [False, True])
def test_main_exits_on_invalid_settings(
    tmp_path: Path, uvicorn_calls: list[Call], reload: bool
) -> None:
    """Invalid settings stop the server before it starts, with or without reload."""
    path = tmp_path / "s.toml"
    path.write_text("[workspace]\n", encoding="utf-8")
    argv = ["--settings", str(path)] + (["--reload"] if reload else [])

    with pytest.raises(SystemExit) as raised:
        entry.main(argv)

    assert raised.value.code == 2
    assert not uvicorn_calls


@pytest.mark.parametrize("reload", [False, True])
def test_main_exits_on_invalid_fit_defaults(
    project_dir: Path, uvicorn_calls: list[Call], reload: bool
) -> None:
    """Invalid ``fit_defaults.toml`` next to the settings stops the server."""
    (project_dir / "fit_defaults.toml").write_text("[pca]\nunknown = 1\n", encoding="utf-8")
    argv = ["--settings", str(project_dir / "settings.toml")] + (
        ["--reload"] if reload else []
    )

    with pytest.raises(SystemExit) as raised:
        entry.main(argv)

    assert raised.value.code == 2
    assert not uvicorn_calls


def test_main_serves_app_object_without_reload(
    project_dir: Path, uvicorn_calls: list[Call]
) -> None:
    """Without ``--reload``, the app object is served and the environment is untouched."""
    entry.main(["--settings", str(project_dir / "settings.toml"), "--port", "8123"])

    [(args, kwargs)] = uvicorn_calls
    assert isinstance(args[0], FastAPI)
    assert kwargs == {"host": "127.0.0.1", "port": 8123}
    assert SETTINGS_ENV not in entry.os.environ


def test_main_serves_factory_with_reload(
    project_dir: Path, uvicorn_calls: list[Call], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``--reload``, the factory is served and receives the absolute settings path."""
    monkeypatch.chdir(project_dir)

    entry.main(["--settings", "settings.toml", "--reload"])

    [(args, kwargs)] = uvicorn_calls
    assert args == (FACTORY_IMPORT_STRING,)
    assert kwargs == {
        "factory": True,
        "reload": True,
        "reload_dirs": [str(entry.PACKAGE_DIR)],
        "host": "127.0.0.1",
        "port": 8000,
    }
    assert entry.PACKAGE_DIR.name == "flat_pca"
    assert entry.os.environ[SETTINGS_ENV] == str(
        (project_dir / "settings.toml").resolve()
    )
