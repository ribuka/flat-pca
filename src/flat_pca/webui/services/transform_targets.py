"""The target files of fit and transform runs, and reuse of transform runs."""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from typing import cast

from .run_dirs import fit_run_reference


def run_target_files(run: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the target files saved in a fit or transform run's configuration.

    Parameters
    ----------
    run : Mapping[str, object]
        A fit or transform run's ``runs`` row.

    Returns
    -------
    list[dict[str, object]]
        Each target file's ``stem``, ``path``, and metadata columns, in the
        form of a catalog row accepted by ``file_entries``.
    """
    config = cast(dict[str, object], json.loads(str(run["config_json"])))
    return [
        {
            "stem": file["stem"],
            "path": file["path"],
            **cast(dict[str, object], file.get("metadata") or {}),
        }
        for file in cast(list[dict[str, object]], config["files"])
    ]


def run_target_stems(run: Mapping[str, object]) -> list[str]:
    """Return the stems of the target files of a fit or transform run.

    Parameters
    ----------
    run : Mapping[str, object]
        A fit or transform run's ``runs`` row.

    Returns
    -------
    list[str]
        Stems in the order of the run's configuration.
    """
    return [str(file["stem"]) for file in run_target_files(run)]


def find_transform_run(
    transform_runs: Iterable[dict[str, object]], fit_run_id: str, stems: Collection[str]
) -> dict[str, object] | None:
    """Return the first succeeded transform run of the same model and targets.

    Parameters
    ----------
    transform_runs : Iterable[dict[str, object]]
        Transform runs to search, newest first.
    fit_run_id : str
        Fit run whose model transforms the targets.
    stems : Collection[str]
        Stems of the transform targets; their order does not matter.

    Returns
    -------
    dict[str, object] | None
        The first succeeded run made with ``fit_run_id`` whose target stems
        are the same set as ``stems``, or ``None``.
    """
    wanted = set(stems)
    for run in transform_runs:
        reference = fit_run_reference(run)
        if (
            run["status"] == "succeeded"
            and reference is not None
            and reference[0] == fit_run_id
            and set(run_target_stems(run)) == wanted
        ):
            return run
    return None
