"""Expansion of path values in the Web UI settings."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"%(\w+)%|\$\{(\w+)\}")
_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


def _expand_environment(value: str) -> str:
    """Replace ``%NAME%`` and ``${NAME}`` with environment variables.

    Parameters
    ----------
    value : str
        Path value.

    Returns
    -------
    str
        Value with environment variables replaced.

    Raises
    ------
    ValueError
        If a referenced environment variable is not set.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ValueError(f"environment variable is not set: {name!r} in {value!r}")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _expand_placeholders(value: str, placeholders: Mapping[str, Path]) -> str:
    """Replace ``{name}`` with settings paths.

    Parameters
    ----------
    value : str
        Path value.
    placeholders : Mapping[str, Path]
        Paths available as placeholders, keyed by name.

    Returns
    -------
    str
        Value with placeholders replaced.

    Raises
    ------
    ValueError
        If a placeholder is not in ``placeholders``.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in placeholders:
            raise ValueError(f"unknown placeholder {{{name}}} in {value!r}")
        return str(placeholders[name])

    return _PLACEHOLDER_PATTERN.sub(replace, value)


def expand_path(value: str, base: Path, placeholders: Mapping[str, Path]) -> Path:
    """Expand one settings path value.

    Environment variables (``%NAME%``, ``${NAME}``) are replaced first, then
    placeholders (``{name}``) and a leading ``~``. A path that is still
    relative is resolved against ``base``.

    Parameters
    ----------
    value : str
        Path value from the settings file.
    base : Path
        Directory relative paths are resolved against.
    placeholders : Mapping[str, Path]
        Paths available as placeholders, keyed by name.

    Returns
    -------
    Path
        Expanded path.

    Raises
    ------
    ValueError
        If an environment variable is not set or a placeholder is unknown.
    """
    expanded = _expand_placeholders(_expand_environment(value), placeholders)
    return base / Path(expanded).expanduser()


def expand_settings_paths(raw: dict[str, Any], base: Path) -> None:
    """Expand ``data.root``, ``workspace.dir``, and ``metadata.csv`` in place.

    ``data.root`` is expanded first and is then available to the other two
    entries as the ``{root}`` placeholder.

    Parameters
    ----------
    raw : dict[str, Any]
        Parsed settings document, modified in place.
    base : Path
        Directory containing the settings file.

    Raises
    ------
    ValueError
        If an environment variable is not set or a placeholder is unknown.
    """
    placeholders: dict[str, Path] = {}
    for section, key, name in (
        ("data", "root", "root"),
        ("workspace", "dir", None),
        ("metadata", "csv", None),
    ):
        table = raw.get(section)
        if not (isinstance(table, dict) and isinstance(table.get(key), str)):
            continue
        path = expand_path(table[key], base, placeholders)
        table[key] = str(path)
        if name is not None:
            placeholders[name] = path
