"""Tests for the target files of runs, the reuse of transform runs, and the transform settings."""

from __future__ import annotations

import json

from flat_pca.webui.services.transform_settings import TransformSettings
from flat_pca.webui.services.transform_targets import (
    find_transform_run,
    run_target_files,
    run_target_stems,
)


def _run(
    run_id: str, stems: list[str], fit_run_id: str | None = None, status: str = "succeeded"
) -> dict[str, object]:
    """Return a ``runs`` row of a fit run, or of a transform run with ``fit_run_id``."""
    config: dict[str, object] = {
        "files": [
            {"stem": stem, "path": f"/data/{stem}.parquet", "metadata": {"lot": "A"}}
            for stem in stems
        ]
    }
    if fit_run_id is not None:
        config |= {"fit_run_id": fit_run_id, "fit_run_dir": f"/runs/{fit_run_id}"}
    return {"run_id": run_id, "status": status, "config_json": json.dumps(config)}


def test_run_target_files_are_catalog_rows_of_the_configured_files() -> None:
    """The configured files come back with their path and metadata columns."""
    run = _run("fit-1", ["b", "a"])

    assert run_target_files(run) == [
        {"stem": "b", "path": "/data/b.parquet", "lot": "A"},
        {"stem": "a", "path": "/data/a.parquet", "lot": "A"},
    ]
    assert run_target_stems(run) == ["b", "a"]


def test_find_transform_run_matches_the_model_and_the_set_of_targets() -> None:
    """The first succeeded run of the same model and the same set of files is found."""
    runs = [
        _run("tr-failed", ["a", "b"], "fit-1", status="failed"),
        _run("tr-other-model", ["a", "b"], "fit-2"),
        _run("tr-other-files", ["a"], "fit-1"),
        _run("tr-new", ["b", "a"], "fit-1"),
        _run("tr-old", ["a", "b"], "fit-1"),
        _run("fit-1", ["a", "b"]),
    ]

    found = find_transform_run(runs, "fit-1", ["a", "b"])

    assert found is not None
    assert found["run_id"] == "tr-new"
    assert find_transform_run(runs, "fit-1", ["a", "b", "c"]) is None
    assert find_transform_run(runs, "fit-3", ["a", "b"]) is None


def test_transform_settings_default_to_the_latest_model_and_the_same_data() -> None:
    """Without a choice, the latest model transforms its own fit targets."""
    settings = TransformSettings()
    assert (settings.model_run_id, settings.use_same_data) == (None, True)

    settings.update("fit-1", use_same_data=False)

    assert (settings.model_run_id, settings.use_same_data) == ("fit-1", False)
