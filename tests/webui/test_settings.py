"""Tests for loading and validating ``settings.toml``."""

from __future__ import annotations

from pathlib import Path

import pytest

from flat_pca.webui.settings import SettingsError, load_settings


def _write(path: Path, text: str) -> Path:
    """Write a settings file and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def test_load_settings_resolves_paths_against_settings_directory(
    project_dir: Path,
) -> None:
    """Relative paths are resolved next to the settings file."""
    settings = load_settings(project_dir / "settings.toml")

    assert settings.workspace.dir == project_dir / "workspace"
    assert settings.data.root == project_dir / "data"
    assert settings.metadata is not None
    assert settings.metadata.csv == project_dir / "meta.csv"
    assert settings.database_path == project_dir / "workspace" / "flatpca.duckdb"
    assert {
        name: column.type for name, column in settings.metadata_columns.items()
    } == {
        "lot": "category",
        "date": "datetime",
        "yield_pct": "number",
    }
    assert settings.ui.heatmap_max_cells == 1_200_000
    assert settings.ui.explore_max_files == 20
    assert settings.jobs.artifact_dtype == "float32"


def test_load_settings_accepts_minimal_file(tmp_path: Path) -> None:
    """Only the workspace and data sections are required."""
    path = _write(tmp_path / "s.toml", '[workspace]\ndir = "w"\n[data]\nroot = "d"\n')

    settings = load_settings(path)

    assert settings.metadata is None
    assert settings.metadata_columns == {}
    assert settings.data.glob == "**/*.parquet"


_BASE = "[workspace]\ndir = 'w'\n[data]\nroot = 'd'\n"
_METADATA = "[metadata]\ncsv = 'm.csv'\nkey = 'k'\n[metadata.columns]\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[data]\nroot = 'd'\n", "workspace"),
        (_BASE + "extra = 1\n", "extra"),
        (_BASE + _METADATA + "lot = { type = 'text' }\n", "type"),
        (
            _BASE + _METADATA + "x = { type = 'number', format = '%Y' }\n",
            "format is only allowed",
        ),
        (_BASE + _METADATA + "stem = { type = 'category' }\n", "reserved"),
        (_BASE + _METADATA + "n_rows = { type = 'number' }\n", "reserved"),
        (_BASE + _METADATA + "path = { type = 'category' }\n", "reserved"),
        (_BASE + _METADATA + "source = { type = 'category' }\n", "reserved"),
        (_BASE + "[ui]\ndefault_color_by = 'lot'\n", "not a metadata column"),
        (_BASE + "[jobs]\nartifact_dtype = 'float16'\n", "artifact_dtype"),
        (_BASE + "[ui]\nheatmap_max_cells = 0\n", "heatmap"),
    ],
)
def test_load_settings_rejects_invalid_settings(
    tmp_path: Path, body: str, message: str
) -> None:
    """Validation errors are reported as ``SettingsError``."""
    path = _write(tmp_path / "s.toml", body)

    with pytest.raises(SettingsError, match=message):
        load_settings(path)


def test_load_settings_rejects_invalid_toml(tmp_path: Path) -> None:
    """A TOML syntax error is reported as ``SettingsError``."""
    path = _write(tmp_path / "s.toml", "[workspace\n")

    with pytest.raises(SettingsError, match="invalid TOML"):
        load_settings(path)


def test_load_settings_rejects_missing_file(tmp_path: Path) -> None:
    """A missing file is reported as ``SettingsError``."""
    with pytest.raises(SettingsError, match="cannot read"):
        load_settings(tmp_path / "missing.toml")


def test_load_settings_reads_local_file_instead(tmp_path: Path) -> None:
    """A sibling ``*.local.toml`` is read in place of the given file."""
    path = _write(
        tmp_path / "settings.toml",
        _BASE + "glob = '*.parquet'\n" + _METADATA + "lot = { type = 'category' }\n",
    )
    _write(
        tmp_path / "settings.local.toml",
        "[workspace]\ndir = 'lw'\n[data]\nroot = 'ld'\n",
    )

    settings = load_settings(path)

    assert settings.workspace.dir == tmp_path / "lw"
    assert settings.data.root == tmp_path / "ld"
    assert settings.data.glob == "**/*.parquet"
    assert settings.metadata is None


def test_load_settings_rejects_invalid_local_file(tmp_path: Path) -> None:
    """A TOML syntax error in the local file names that file."""
    path = _write(tmp_path / "settings.toml", _BASE)
    _write(tmp_path / "settings.local.toml", "[data\n")

    with pytest.raises(SettingsError, match=r"invalid TOML in .*settings\.local\.toml"):
        load_settings(path)


def test_load_settings_reads_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a path, ``config/settings.toml`` under the cwd is read."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(config_dir / "settings.toml", _BASE)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.workspace.dir == config_dir / "w"


def test_load_settings_expands_root_placeholder(tmp_path: Path) -> None:
    """``{root}`` in ``workspace.dir`` and ``metadata.csv`` is ``data.root``."""
    body = (
        "[workspace]\ndir = '{root}/w'\n[data]\nroot = 'd'\n"
        "[metadata]\ncsv = '{root}/meta.csv'\nkey = 'k'\n"
    )
    settings = load_settings(_write(tmp_path / "s.toml", body))

    assert settings.workspace.dir == tmp_path / "d" / "w"
    assert settings.metadata is not None
    assert settings.metadata.csv == tmp_path / "d" / "meta.csv"


@pytest.mark.parametrize("reference", ["%FLATPCA_TEST_DIR%", "${FLATPCA_TEST_DIR}"])
def test_load_settings_expands_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    """``%NAME%`` and ``${NAME}`` are replaced with environment variables."""
    monkeypatch.setenv("FLATPCA_TEST_DIR", str(tmp_path / "env"))
    body = f"[workspace]\ndir = 'w'\n[data]\nroot = '{reference}/d'\n"

    settings = load_settings(_write(tmp_path / "s.toml", body))

    assert settings.data.root == tmp_path / "env" / "d"


def test_load_settings_expands_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leading ``~`` is the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    body = "[workspace]\ndir = 'w'\n[data]\nroot = '~/d'\n"

    settings = load_settings(_write(tmp_path / "s.toml", body))

    assert settings.data.root == tmp_path / "home" / "d"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "[workspace]\ndir = 'w'\n[data]\nroot = '%FLATPCA_UNSET%'\n",
            "environment variable is not set: 'FLATPCA_UNSET'",
        ),
        (
            "[workspace]\ndir = '{data}/w'\n[data]\nroot = 'd'\n",
            r"unknown placeholder \{data\}",
        ),
        (
            "[workspace]\ndir = '{data.root}/w'\n[data]\nroot = 'd'\n",
            r"unknown placeholder \{data\.root\}",
        ),
        (
            "[workspace]\ndir = '{unknown-name}/w'\n[data]\nroot = 'd'\n",
            r"unknown placeholder \{unknown-name\}",
        ),
        (
            "[workspace]\ndir = '{}/w'\n[data]\nroot = 'd'\n",
            r"unknown placeholder \{\}",
        ),
        (
            "[workspace]\ndir = 'w'\n[data]\nroot = '{root}/d'\n",
            r"unknown placeholder \{root\}",
        ),
    ],
)
def test_load_settings_rejects_unexpandable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, message: str
) -> None:
    """Unset environment variables and unknown placeholders are errors."""
    monkeypatch.delenv("FLATPCA_UNSET", raising=False)
    path = _write(tmp_path / "s.toml", body)

    with pytest.raises(SettingsError, match=message):
        load_settings(path)
