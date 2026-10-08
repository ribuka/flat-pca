"""Reading and validating the artifacts of a fit run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from flat_pca.feature_engineering.pca import PcaModel
from flat_pca.spectral.schema import SOURCE_COLUMN

from ..jobs.fit_run import (
    COMPONENTS_FILE,
    FEATURES_FILE,
    PCA_STATE_FILE,
    SAMPLES_FILE,
    SCORES_FILE,
    X_FILE,
)


class RunArtifactError(ValueError):
    """Raised when the artifacts of a run cannot be read.

    Artifacts carry no format version, so artifacts written by an older
    version of the application are reported this way and the run must be
    executed again.
    """


@dataclass(frozen=True)
class FitArtifacts:
    """Artifacts of one fit run.

    Attributes
    ----------
    features : pl.DataFrame
        One row per feature, in ``x`` column order.
    samples : pl.DataFrame
        One row per file, in ``x`` row order.
    scores : pl.DataFrame
        Scores, T², and Q of the files kept by the imputation stage.
    x : np.ndarray
        Preprocessed matrix before imputation, memory-mapped read-only in
        its saved dtype. Convert rows to ``float64`` before computing.
    model : PcaModel
        PCA pipeline restored in ``float64``.
    """

    features: pl.DataFrame
    samples: pl.DataFrame
    scores: pl.DataFrame
    x: np.ndarray
    model: PcaModel


@dataclass(frozen=True)
class DisplayArtifacts:
    """Artifacts of one fit run needed to display its matrices.

    Unlike ``FitArtifacts``, no PCA model is restored, so the components
    stay memory-mapped and only the rows a view needs are read.

    Attributes
    ----------
    features : pl.DataFrame
        One row per feature, in ``x`` and ``components`` column order.
    samples : pl.DataFrame
        One row per file, in ``x`` row order.
    scores : pl.DataFrame
        Scores, T², and Q of the files kept by the imputation stage.
    x : np.ndarray
        Preprocessed matrix before imputation, memory-mapped read-only in
        its saved dtype.
    components : np.ndarray
        ``pca.components_`` (``n_comp × n_features``), memory-mapped
        read-only in its saved dtype.
    """

    features: pl.DataFrame
    samples: pl.DataFrame
    scores: pl.DataFrame
    x: np.ndarray
    components: np.ndarray


_READ_ERRORS = (OSError, KeyError, ValueError, pl.exceptions.PolarsError)


def artifact_error(error: Exception) -> RunArtifactError:
    """Return the error reported for unreadable artifacts.

    Parameters
    ----------
    error : Exception
        Error raised while reading or validating the artifacts.

    Returns
    -------
    RunArtifactError
        Error asking to execute the run again.
    """
    return RunArtifactError(
        f"Cannot read the run's artifacts; they are outdated or broken, so run the fit again "
        f"({type(error).__name__}: {error})."
    )


def _check_score_sources(scores: pl.DataFrame, samples: pl.DataFrame) -> None:
    """Check that the scores belong to the run's samples.

    Parameters
    ----------
    scores : pl.DataFrame
        The run's ``scores.parquet``.
    samples : pl.DataFrame
        The run's ``samples.parquet``.

    Raises
    ------
    ValueError
        If the scores lack the ``source`` column or hold unknown sources.
    """
    if SOURCE_COLUMN not in scores.columns:
        raise ValueError(f"{SCORES_FILE} lacks columns {[SOURCE_COLUMN]}")
    unknown = set(scores[SOURCE_COLUMN]) - set(samples[SOURCE_COLUMN])
    if unknown:
        raise ValueError(f"{SCORES_FILE} holds unknown sources {sorted(unknown)}")


def load_display_artifacts(run_dir: Path) -> DisplayArtifacts:
    """Load and validate the artifacts a fit run's display needs.

    Parameters
    ----------
    run_dir : Path
        Run directory written by ``run_fit``.

    Returns
    -------
    DisplayArtifacts
        The artifacts, with ``X.npy`` and ``components.npy`` memory-mapped.

    Raises
    ------
    RunArtifactError
        If a file is missing, unreadable, or inconsistent with the others.
    """
    try:
        artifacts = DisplayArtifacts(
            features=pl.read_parquet(run_dir / FEATURES_FILE),
            samples=pl.read_parquet(run_dir / SAMPLES_FILE),
            scores=pl.read_parquet(run_dir / SCORES_FILE),
            x=np.load(run_dir / X_FILE, mmap_mode="r"),
            components=np.load(run_dir / COMPONENTS_FILE, mmap_mode="r"),
        )
        n_features = artifacts.features.height
        expected = (artifacts.samples.height, n_features)
        if artifacts.x.shape != expected:
            raise ValueError(f"{X_FILE} is shaped {artifacts.x.shape}, expected {expected}")
        components = artifacts.components
        if components.ndim != 2 or components.shape[0] < 1 or components.shape[1] != n_features:
            raise ValueError(
                f"{COMPONENTS_FILE} is shaped {components.shape}, "
                f"expected (n_components, {n_features})"
            )
        _check_score_sources(artifacts.scores, artifacts.samples)
    except _READ_ERRORS as error:
        raise artifact_error(error) from error
    return artifacts


def _read_pca_model(run_dir: Path, features: pl.DataFrame) -> PcaModel:
    """Restore the PCA pipeline of a fit run.

    Parameters
    ----------
    run_dir : Path
        Run directory written by ``run_fit``.
    features : pl.DataFrame
        The run's ``features.parquet``.

    Returns
    -------
    PcaModel
        PCA pipeline restored in ``float64``.

    Raises
    ------
    OSError, KeyError, ValueError
        If a file is missing, unreadable, or inconsistent with the features.
    """
    components = np.load(run_dir / COMPONENTS_FILE, mmap_mode="r")
    with np.load(run_dir / PCA_STATE_FILE) as state:
        return PcaModel.from_pca_state(features["feature"].to_list(), components, state)


def load_pca_model(run_dir: Path) -> PcaModel:
    """Load the PCA pipeline of a fit run.

    Unlike ``load_display_artifacts``, the components are read into memory
    in ``float64``, so load the model only for views that reconstruct
    features.

    Parameters
    ----------
    run_dir : Path
        Run directory written by ``run_fit``.

    Returns
    -------
    PcaModel
        PCA pipeline restored in ``float64``, with ``columns`` in
        ``features.parquet`` order.

    Raises
    ------
    RunArtifactError
        If a file is missing, unreadable, or inconsistent with the others.
    """
    try:
        return _read_pca_model(run_dir, pl.read_parquet(run_dir / FEATURES_FILE))
    except _READ_ERRORS as error:
        raise artifact_error(error) from error


def _check_shapes(artifacts: FitArtifacts) -> None:
    """Check that the artifacts describe the same samples and features.

    Parameters
    ----------
    artifacts : FitArtifacts
        Loaded artifacts.

    Raises
    ------
    ValueError
        If the shapes, columns, or sources disagree.
    """
    expected = (artifacts.samples.height, artifacts.features.height)
    if artifacts.x.shape != expected:
        raise ValueError(f"{X_FILE} is shaped {artifacts.x.shape}, expected {expected}")
    if list(artifacts.model.columns) != artifacts.features["feature"].to_list():
        raise ValueError(f"{PCA_STATE_FILE} does not match {FEATURES_FILE}")
    missing = [
        column
        for column in (SOURCE_COLUMN, *artifacts.model.pca_column_names)
        if column not in artifacts.scores.columns
    ]
    if missing:
        raise ValueError(f"{SCORES_FILE} lacks columns {missing}")
    _check_score_sources(artifacts.scores, artifacts.samples)


def load_fit_artifacts(run_dir: Path) -> FitArtifacts:
    """Load and validate the artifacts of a fit run.

    Parameters
    ----------
    run_dir : Path
        Run directory written by ``run_fit``.

    Returns
    -------
    FitArtifacts
        The artifacts, with ``X.npy`` memory-mapped.

    Raises
    ------
    RunArtifactError
        If a file is missing, unreadable, or inconsistent with the others,
        for example because it was written in an older format.
    """
    try:
        features = pl.read_parquet(run_dir / FEATURES_FILE)
        model = _read_pca_model(run_dir, features)
        artifacts = FitArtifacts(
            features=features,
            samples=pl.read_parquet(run_dir / SAMPLES_FILE),
            scores=pl.read_parquet(run_dir / SCORES_FILE),
            x=np.load(run_dir / X_FILE, mmap_mode="r"),
            model=model,
        )
        _check_shapes(artifacts)
    except _READ_ERRORS as error:
        raise artifact_error(error) from error
    return artifacts


def register_fit_result(config: dict[str, object], run_dir: Path) -> dict[str, object]:
    """Validate the artifacts of a finished fit job.

    Used as the job's ``finalize`` step in the app process.

    Parameters
    ----------
    config : dict[str, object]
        Job configuration built by ``build_fit_config``.
    run_dir : Path
        Run directory written by ``run_fit``.

    Returns
    -------
    dict[str, object]
        ``runs`` columns to store: file, feature, and component counts.

    Raises
    ------
    RunArtifactError
        If the artifacts cannot be read or are inconsistent.
    ValueError
        If the files differ from the configured ones.
    """
    artifacts = load_fit_artifacts(run_dir)
    stems = sorted(str(file["stem"]) for file in config["files"])  # type: ignore[union-attr]
    if sorted(artifacts.samples["stem"].to_list()) != stems:
        raise ValueError(f"{SAMPLES_FILE} does not list the configured files")
    return {
        "n_files": artifacts.samples.height,
        "n_features": artifacts.features.height,
        "n_components": artifacts.model.n_component,
    }
