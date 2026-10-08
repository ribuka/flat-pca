"""Succeeded fit and transform runs of the synthetic spectra for the exploration tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from preprocess_settings import preprocess_settings

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.fit_run import build_fit_config, run_fit
from flat_pca.webui.jobs.transform_run import build_transform_config, run_transform
from flat_pca.webui.services.runs import get_run, insert_run, update_run
from flat_pca.webui.services.transform_artifacts import register_transform_result
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import FIT_JOB, TRANSFORM_JOB

STATISTICS = {"cumulative_explained_variance": 2, "alpha": 0.01}


def register_fit_run(
    database: Database,
    settings: Settings,
    spectra_paths: list[Path],
    run_id: str,
    impute_strategy: str,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
    scaling_strategy: str = "z-score",
) -> str:
    """Execute and register a succeeded fit run of the synthetic spectra.

    The workspace fixture's own files are too few to fit, so the run is
    executed directly on the synthetic spectra. It uses the ``sqrt``
    intensity transform with scale 2 and three components.

    Parameters
    ----------
    database : Database
        Workspace database to register the run in.
    settings : Settings
        Application settings.
    spectra_paths : list[Path]
        Synthetic spectra files (see ``spectra.write_spectra``).
    run_id : str
        Identifier of the run.
    impute_strategy : str
        Missing-value handling of the fit.
    metadata : Mapping[str, Mapping[str, object]] | None, default None
        Metadata column values keyed by stem; without them every metadata
        value is missing.
    scaling_strategy : str, default "z-score"
        Feature scaling of the fit.

    Returns
    -------
    str
        The run's identifier.
    """
    config = build_fit_config(
        settings,
        [
            {"stem": path.stem, "path": str(path), **(metadata or {}).get(path.stem, {})}
            for path in spectra_paths
        ],
        preprocess_settings(intensity_transform="sqrt", intensity_transform_scale=2.0),
        {
            "n_component": 3,
            "impute_strategy": impute_strategy,
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": scaling_strategy,
        },
        STATISTICS,
        STATISTICS,
    )
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True)
    run_fit(json.loads(json.dumps(config)), run_dir)
    insert_run(database, run_id, FIT_JOB, config, run_dir)
    update_run(database, run_id, status="succeeded")
    return run_id


def register_transform_run(
    database: Database,
    settings: Settings,
    fit_run_id: str,
    paths: list[Path],
    run_id: str,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Execute and register a succeeded transform run with a fit run's model.

    Parameters
    ----------
    database : Database
        Workspace database holding the fit run.
    settings : Settings
        Application settings.
    fit_run_id : str
        Succeeded fit run whose model transforms the files.
    paths : list[Path]
        Spectra files to transform.
    run_id : str
        Identifier of the run.
    metadata : Mapping[str, Mapping[str, object]] | None, default None
        Metadata column values keyed by stem; without them every metadata
        value is missing.

    Returns
    -------
    str
        The run's identifier.
    """
    fit_run = get_run(database, fit_run_id)
    assert fit_run is not None
    config = build_transform_config(
        settings,
        [
            {"stem": path.stem, "path": str(path), **(metadata or {}).get(path.stem, {})}
            for path in paths
        ],
        fit_run,
    )
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True)
    config = json.loads(json.dumps(config))
    run_transform(config, run_dir)
    insert_run(database, run_id, TRANSFORM_JOB, config, run_dir)
    update_run(
        database,
        run_id,
        status="succeeded",
        **register_transform_result(config, run_dir),
    )
    return run_id


def register_shown_run(
    database: Database,
    settings: Settings,
    spectra_paths: list[Path],
    run_id: str,
    impute_strategy: str,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
    scaling_strategy: str = "z-score",
) -> str:
    """Register a fit run and a transform run of its own files for the display screens.

    A fit run holds no scores, so the display screens show a transform run.
    The fit run is registered as ``{run_id}-fit`` (see ``register_fit_run``)
    and the transform run of the same files as ``run_id``, so
    ``settings.runs_dir / run_id`` holds the shown ``samples.parquet``,
    ``X.npy``, and ``scores.parquet``.

    Parameters
    ----------
    database : Database
        Workspace database to register the runs in.
    settings : Settings
        Application settings.
    spectra_paths : list[Path]
        Synthetic spectra files (see ``spectra.write_spectra``).
    run_id : str
        Identifier of the transform run.
    impute_strategy : str
        Missing-value handling of the fit.
    metadata : Mapping[str, Mapping[str, object]] | None, default None
        Metadata column values keyed by stem.
    scaling_strategy : str, default "z-score"
        Feature scaling of the fit.

    Returns
    -------
    str
        The transform run's identifier.
    """
    fit_run_id = register_fit_run(
        database,
        settings,
        spectra_paths,
        f"{run_id}-fit",
        impute_strategy,
        metadata,
        scaling_strategy,
    )
    return register_transform_run(
        database, settings, fit_run_id, spectra_paths, run_id, metadata
    )
