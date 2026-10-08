"""Directories holding the model and the data a fit or transform run shows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

FIT_RUN_ID_KEY = "fit_run_id"
FIT_RUN_DIR_KEY = "fit_run_dir"


@dataclass(frozen=True)
class RunDirs:
    """Run directories of the model and of the transformed data.

    Attributes
    ----------
    model : Path
        Fit run directory holding ``features.parquet``, ``components.npy``,
        and ``pca_state.npz``.
    data : Path
        Run directory holding ``samples.parquet``, ``X.npy``, and
        ``scores.parquet``: the fit run's own for a fit run, the transform
        run's for a transform run.
    """

    model: Path
    data: Path


def fit_run_reference(run: Mapping[str, object]) -> tuple[str, Path] | None:
    """Return the fit run a transform run was transformed with.

    Parameters
    ----------
    run : Mapping[str, object]
        A ``runs`` row.

    Returns
    -------
    tuple[str, Path] | None
        The fit run's identifier and directory, saved in a transform run's
        configuration; ``None`` for any other run.
    """
    config = json.loads(str(run["config_json"]))
    if not isinstance(config, dict) or FIT_RUN_DIR_KEY not in config:
        return None
    return str(config[FIT_RUN_ID_KEY]), Path(str(config[FIT_RUN_DIR_KEY]))


def run_dirs(run: Mapping[str, object]) -> RunDirs:
    """Return the directories of the model and the data a run shows.

    Parameters
    ----------
    run : Mapping[str, object]
        A fit or transform run's ``runs`` row.

    Returns
    -------
    RunDirs
        For a fit run, its own directory twice; for a transform run, its
        fit run's directory and its own.
    """
    data = Path(str(run["artifact_dir"]))
    reference = fit_run_reference(run)
    return RunDirs(model=data if reference is None else reference[1], data=data)
