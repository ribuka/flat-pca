"""Locating and reading the Web UI settings TOML file."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_PATH = Path("config") / "settings.toml"


def local_settings_path(path: Path) -> Path:
    """Return the local settings file path for a settings file.

    Parameters
    ----------
    path : Path
        Settings file path, e.g. ``config/settings.toml``.

    Returns
    -------
    Path
        Sibling ``<stem>.local<suffix>`` path, e.g. ``config/settings.local.toml``.
    """
    return path.with_name(f"{path.stem}.local{path.suffix}")


def select_settings_file(path: Path) -> Path:
    """Return the settings file to read in place of ``path``.

    Parameters
    ----------
    path : Path
        Settings file path.

    Returns
    -------
    Path
        :func:`local_settings_path` when that file exists, otherwise ``path``.
    """
    local_path = local_settings_path(path)
    return local_path if local_path.is_file() else path


def read_settings_document(path: Path) -> dict[str, Any]:
    """Read one settings TOML file.

    Parameters
    ----------
    path : Path
        TOML file path.

    Returns
    -------
    dict[str, Any]
        Parsed document.

    Raises
    ------
    ValueError
        If the file cannot be read or is not valid TOML.
    """
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except OSError as error:
        raise ValueError(f"cannot read settings file {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {path}: {error}") from error
