"""The target files of fit and transform runs, and reuse of transform runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import cast

from .run_dirs import FIT_RUN_ID_KEY, fit_run_reference


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


FileIdentity = tuple[str, str, object, object]


def file_identities(files: Iterable[Mapping[str, object]]) -> frozenset[FileIdentity] | None:
    """Return the identities of the target files of a transform configuration.

    Parameters
    ----------
    files : Iterable[Mapping[str, object]]
        ``files`` entries of a transform configuration, each with ``stem``,
        ``path``, ``size``, and ``mtime_ns`` (see ``build_transform_config``).

    Returns
    -------
    frozenset[FileIdentity] | None
        ``(stem, path, size, mtime_ns)`` of every file, or ``None`` if a
        file's size or modification time is unknown, so that it matches no
        other run.
    """
    identities = set()
    for file in files:
        size, mtime_ns = file.get("size"), file.get("mtime_ns")
        if size is None or mtime_ns is None:
            return None
        identities.add((str(file["stem"]), str(file["path"]), size, mtime_ns))
    return frozenset(identities)


def find_transform_run(
    transform_runs: Iterable[dict[str, object]], config: Mapping[str, object]
) -> dict[str, object] | None:
    """Return the first succeeded transform run of the same model and target files.

    Parameters
    ----------
    transform_runs : Iterable[dict[str, object]]
        Transform runs to search, newest first.
    config : Mapping[str, object]
        Configuration of the transform to run (see ``build_transform_config``).

    Returns
    -------
    dict[str, object] | None
        The first succeeded run made with the same fit run whose target
        files are the same set (order aside) with the same paths, sizes,
        and modification times, or ``None``. A file changed since a run
        makes that run unusable.
    """
    wanted = file_identities(cast(list[dict[str, object]], config["files"]))
    if wanted is None:
        return None
    fit_run_id = str(config[FIT_RUN_ID_KEY])
    for run in transform_runs:
        reference = fit_run_reference(run)
        if run["status"] != "succeeded" or reference is None or reference[0] != fit_run_id:
            continue
        saved = cast(dict[str, object], json.loads(str(run["config_json"])))
        if file_identities(cast(list[dict[str, object]], saved["files"])) == wanted:
            return run
    return None
