"""Validating the artifacts of a finished transform run."""

from __future__ import annotations

from pathlib import Path

from ..run_layout import FIT_RUN_DIR_KEY, SAMPLES_FILE
from .fit_artifacts import load_display_artifacts
from .run_dirs import RunDirs


def register_transform_result(config: dict[str, object], run_dir: Path) -> dict[str, object]:
    """Validate the artifacts of a finished transform job.

    Used as the job's ``finalize`` step in the app process.

    Parameters
    ----------
    config : dict[str, object]
        Job configuration built by ``build_transform_config``.
    run_dir : Path
        Transform run directory written by ``run_transform``.

    Returns
    -------
    dict[str, object]
        ``runs`` columns to store: file, feature, and component counts.

    Raises
    ------
    RunArtifactError
        If the artifacts of the transform run or its fit run cannot be read
        or are inconsistent.
    ValueError
        If the files differ from the configured ones.
    """
    artifacts = load_display_artifacts(
        RunDirs(model=Path(str(config[FIT_RUN_DIR_KEY])), data=run_dir)
    )
    stems = sorted(str(file["stem"]) for file in config["files"])  # type: ignore[union-attr]
    if sorted(artifacts.samples["stem"].to_list()) != stems:
        raise ValueError(f"{SAMPLES_FILE} does not list the configured files")
    return {
        "n_files": artifacts.samples.height,
        "n_features": artifacts.features.height,
        "n_components": artifacts.components.shape[0],
    }
