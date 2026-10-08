"""Tests for the target files of runs, the reuse of transform runs, and the transform settings."""

from __future__ import annotations

import json

from flat_pca.webui.services.transform_settings import TransformSettings
from flat_pca.webui.services.transform_targets import (
    find_transform_run,
    run_target_files,
    run_target_stems,
)


def _files(stems: list[str], mtime_ns: int = 1) -> list[dict[str, object]]:
    """Return ``files`` entries of a configuration, each with its file state."""
    return [
        {
            "stem": stem,
            "path": f"/data/{stem}.parquet",
            "metadata": {"lot": "A"},
            "size": 100,
            "mtime_ns": mtime_ns,
        }
        for stem in stems
    ]


def _config(fit_run_id: str, files: list[dict[str, object]]) -> dict[str, object]:
    """Return a transform configuration of ``files`` with the model of ``fit_run_id``."""
    return {"fit_run_id": fit_run_id, "fit_run_dir": f"/runs/{fit_run_id}", "files": files}


def _run(run_id: str, config: dict[str, object], status: str = "succeeded") -> dict[str, object]:
    """Return a ``runs`` row with ``config``."""
    return {"run_id": run_id, "status": status, "config_json": json.dumps(config)}


def test_run_target_files_are_catalog_rows_of_the_configured_files() -> None:
    """The configured files come back with their path and metadata columns."""
    run = _run("fit-1", {"files": _files(["b", "a"])})

    assert [
        {key: file[key] for key in ("stem", "path", "lot")} for file in run_target_files(run)
    ] == [
        {"stem": "b", "path": "/data/b.parquet", "lot": "A"},
        {"stem": "a", "path": "/data/a.parquet", "lot": "A"},
    ]
    assert run_target_stems(run) == ["b", "a"]


def test_find_transform_run_matches_the_model_and_the_unchanged_targets() -> None:
    """The first succeeded run of the same model and the same unchanged files is found."""
    runs = [
        _run("tr-failed", _config("fit-1", _files(["a", "b"])), status="failed"),
        _run("tr-other-model", _config("fit-2", _files(["a", "b"]))),
        _run("tr-other-files", _config("fit-1", _files(["a"]))),
        _run("tr-new", _config("fit-1", _files(["b", "a"]))),
        _run("tr-old", _config("fit-1", _files(["a", "b"]))),
        _run("fit-1", {"files": _files(["a", "b"])}),
    ]

    found = find_transform_run(runs, _config("fit-1", _files(["a", "b"])))

    assert found is not None
    assert found["run_id"] == "tr-new"
    assert find_transform_run(runs, _config("fit-1", _files(["a", "b", "c"]))) is None
    assert find_transform_run(runs, _config("fit-3", _files(["a", "b"]))) is None


def test_find_transform_run_skips_runs_of_changed_or_unreadable_files() -> None:
    """A file modified since a run, or one that cannot be read, matches no run."""
    runs = [_run("tr-1", _config("fit-1", _files(["a", "b"])))]
    unreadable = _files(["a", "b"])
    unreadable[1] |= {"size": None, "mtime_ns": None}

    assert find_transform_run(runs, _config("fit-1", _files(["a", "b"], mtime_ns=2))) is None
    assert find_transform_run(runs, _config("fit-1", unreadable)) is None


def test_transform_settings_default_to_the_latest_model_and_the_same_data() -> None:
    """Without a choice, the latest model transforms its own fit targets."""
    settings = TransformSettings()
    assert (settings.model_run_id, settings.use_same_data) == (None, True)

    assert settings.update("fit-1", use_same_data=False) is True

    assert (settings.model_run_id, settings.use_same_data) == ("fit-1", False)
    assert settings.update("fit-1", use_same_data=True) is False
