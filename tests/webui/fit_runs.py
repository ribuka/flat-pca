"""Succeeded fit runs of the synthetic spectra for the exploration tests."""

from __future__ import annotations

import json
from pathlib import Path

from flat_pca.webui.database import Database
from flat_pca.webui.jobs.fit_run import build_fit_config, run_fit
from flat_pca.webui.services.runs import insert_run, update_run
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import FIT_JOB

STATISTICS = {"cumulative_explained_variance": 2, "alpha": 0.01}


def register_fit_run(
    database: Database,
    settings: Settings,
    spectra_paths: list[Path],
    run_id: str,
    impute_strategy: str,
) -> str:
    """Execute and register a succeeded fit run of the synthetic spectra.

    The workspace fixture's own files are too few to fit, so the run is
    executed directly on the synthetic spectra. It uses the ``sqrt``
    intensity transform with scale 2, ``z-score`` scaling, and three
    components.

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

    Returns
    -------
    str
        The run's identifier.
    """
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in spectra_paths],
        {
            "target_steps": [1, 2],
            "max_null_ratio": 0.1,
            "intensity_transform": "sqrt",
            "intensity_transform_scale": 2.0,
        },
        {
            "n_component": 3,
            "impute_strategy": impute_strategy,
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "z-score",
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
