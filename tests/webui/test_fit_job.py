"""Tests for the fit job, calling ``run_fit`` without a child process."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from spectra import SPECTRA_FILE_COUNT, SPECTRA_SHORT_FILE, SPECTRA_WAVELENGTHS

from flat_pca.feature_engineering.flatten_pca import preprocess_and_flatten
from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig, transform_pca
from flat_pca.webui.jobs.fit_run import (
    AUTO_COMPONENT_CUMULATIVE,
    COMPONENTS_FILE,
    FEATURES_FILE,
    PCA_STATE_FILE,
    SAMPLES_FILE,
    SCORES_FILE,
    X_FILE,
    build_fit_config,
    run_fit,
)
from flat_pca.webui.jobs.progress import read_progress
from flat_pca.webui.services.fit_artifacts import (
    RunArtifactError,
    load_fit_artifacts,
    register_fit_result,
)
from flat_pca.webui.settings import Settings

STATISTICS = {"cumulative_explained_variance": 2, "alpha": 0.01}


def _fit_config(
    settings: Settings,
    paths: list[Path],
    *,
    preprocess: dict[str, object] | None = None,
    **pca: object,
) -> dict[str, object]:
    """Build a fit configuration for the synthetic spectra.

    Even-indexed files get ``lot`` A with a date and yield; odd-indexed
    files have no metadata.
    """
    files = [
        {
            "stem": path.stem,
            "path": str(path),
            "lot": "A" if index % 2 == 0 else None,
            "date": datetime.fromisoformat(f"2026-01-{index + 1:02d}T08:00:00")
            if index % 2 == 0
            else None,
            "yield_pct": float(index) if index % 2 == 0 else None,
        }
        for index, path in enumerate(paths)
    ]
    return build_fit_config(
        settings,
        files,
        {"target_steps": [1, 2], "max_null_ratio": 0.1, **(preprocess or {})},
        {
            "n_component": None,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "none",
            **pca,
        },
        STATISTICS,
        STATISTICS,
    )


def _run(config: dict[str, object], run_dir: Path) -> Path:
    """Run the job through a JSON round trip, as the executor does."""
    run_dir.mkdir()
    run_fit(json.loads(json.dumps(config)), run_dir)
    return run_dir


def test_run_fit_writes_artifacts_in_flatten_order(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Artifacts follow the flattened row and column order."""
    run_dir = _run(_fit_config(settings, spectra_paths), tmp_path / "run")
    flattened = preprocess_and_flatten(spectra_paths, target_steps=[1, 2]).collect()
    feature_names = flattened.columns[1:]

    features = pl.read_parquet(run_dir / FEATURES_FILE)
    assert features.columns == ["feature", "wavelength", "Step", "Sequence", "StepTime"]
    assert features["feature"].to_list() == feature_names
    assert features.height == 7 * len(SPECTRA_WAVELENGTHS)
    assert features.row(0) == (feature_names[0], 400.0, 1, 1, 0.0)

    samples = pl.read_parquet(run_dir / SAMPLES_FILE)
    assert samples.columns == ["source", "stem", "lot", "date", "yield_pct"]
    assert samples["source"].to_list() == flattened["source"].to_list()
    assert samples["stem"].to_list() == [path.stem for path in spectra_paths]
    assert samples.schema["date"] == pl.Datetime("us")
    assert samples.row(0)[2:] == ("A", datetime.fromisoformat("2026-01-01T08:00:00"), 0.0)
    assert samples.row(1)[2:] == (None, None, None)

    x = np.load(run_dir / X_FILE)
    assert x.dtype == np.float32
    assert x.shape == (SPECTRA_FILE_COUNT, len(feature_names))
    np.testing.assert_allclose(
        x, flattened.select(feature_names).to_numpy(), rtol=1e-6
    )
    assert np.isnan(x).sum() == len(SPECTRA_WAVELENGTHS)
    assert np.isnan(x[SPECTRA_SHORT_FILE]).sum() == len(SPECTRA_WAVELENGTHS)

    components = np.load(run_dir / COMPONENTS_FILE)
    assert components.dtype == np.float32
    assert components.shape[1] == len(feature_names)

    scores = pl.read_parquet(run_dir / SCORES_FILE)
    n_component = components.shape[0]
    assert scores.columns == [
        "source",
        *(f"pca-{index}" for index in range(1, n_component + 1)),
        *MahalanobisConfig().column_names,
        *SpeConfig().column_names,
    ]
    assert scores["source"].to_list() == samples["source"].to_list()

    with np.load(run_dir / PCA_STATE_FILE) as state:
        assert int(state["n_component"]) == n_component
        assert state["impute_values"].shape == (len(feature_names),)
    progress = read_progress(run_dir)
    assert progress is not None
    assert (progress.stage, progress.done, progress.total) == ("save", 4, 4)


@pytest.mark.parametrize("impute_strategy", ["drop", "median", "kmeans"])
@pytest.mark.parametrize("scaling_strategy", ["none", "pareto"])
def test_restored_model_reproduces_saved_scores(
    settings: Settings,
    spectra_paths: list[Path],
    tmp_path: Path,
    impute_strategy: str,
    scaling_strategy: str,
) -> None:
    """Scores, T², and Q recomputed from the artifacts match ``scores.parquet``."""
    run_dir = _run(
        _fit_config(
            settings,
            spectra_paths,
            impute_strategy=impute_strategy,
            scaling_strategy=scaling_strategy,
        ),
        tmp_path / "run",
    )

    artifacts = load_fit_artifacts(run_dir)
    frame = pl.DataFrame(
        np.asarray(artifacts.x, dtype=np.float64),
        schema=artifacts.features["feature"].to_list(),
    ).insert_column(0, artifacts.samples["source"])
    recomputed = transform_pca(
        frame.lazy(),
        artifacts.model,
        mahalanobis=MahalanobisConfig(**STATISTICS),
        spe=SpeConfig(**STATISTICS),
    ).collect()

    expected_rows = SPECTRA_FILE_COUNT - (impute_strategy == "drop")
    assert artifacts.scores.height == expected_rows
    assert recomputed["source"].to_list() == artifacts.scores["source"].to_list()
    columns = [
        *artifacts.model.pca_column_names,
        "mahalanobis_sq",
        "mahalanobis_ucl",
        "spe",
        "spe_ucl",
    ]
    np.testing.assert_allclose(
        recomputed.select(columns).to_numpy(),
        artifacts.scores.select(columns).to_numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    assert artifacts.model.impute_model.strategy == impute_strategy
    assert artifacts.model.scaling_model.strategy == scaling_strategy


def test_automatic_component_count_reaches_the_cumulative_ratio(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Without ``n_component``, the fewest components reaching 0.99 are kept."""
    run_dir = _run(_fit_config(settings, spectra_paths), tmp_path / "run")

    model = load_fit_artifacts(run_dir).model
    cumulative = np.cumsum(model.pca.explained_variance_ratio_)
    assert cumulative[-1] >= AUTO_COMPONENT_CUMULATIVE
    assert model.n_component == 1 or cumulative[-2] < AUTO_COMPONENT_CUMULATIVE
    assert model.n_component < SPECTRA_FILE_COUNT


def test_explicit_component_count_and_float64_artifacts(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """An explicit ``n_component`` is kept, and float64 saves float64 matrices."""
    config = _fit_config(settings, spectra_paths, n_component=4)
    config["artifact_dtype"] = "float64"

    run_dir = _run(config, tmp_path / "run")

    assert np.load(run_dir / X_FILE).dtype == np.float64
    assert np.load(run_dir / COMPONENTS_FILE).shape[0] == 4
    assert register_fit_result(config, run_dir) == {
        "n_files": SPECTRA_FILE_COUNT,
        "n_features": 7 * len(SPECTRA_WAVELENGTHS),
        "n_components": 4,
    }


def test_log1p_domain_error_fails_the_job(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """``log1p`` of a value at or below -scale raises ``ValueError``."""
    negative = pl.read_parquet(spectra_paths[0]).with_columns(
        pl.col(f"{wavelength:.1f}nm") * -1.0 for wavelength in SPECTRA_WAVELENGTHS
    )
    negative.write_parquet(spectra_paths[0])
    config = _fit_config(
        settings,
        spectra_paths,
        preprocess={"intensity_transform": "log1p", "intensity_transform_scale": 1.0},
    )

    with pytest.raises(ValueError, match="log1p"):
        _run(config, tmp_path / "run")


def test_register_rejects_missing_artifacts(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """A missing artifact fails the run with a message asking to rerun it."""
    config = _fit_config(settings, spectra_paths)
    run_dir = _run(config, tmp_path / "run")
    (run_dir / SCORES_FILE).unlink()

    with pytest.raises(RunArtifactError, match="run the fit again"):
        register_fit_result(config, run_dir)


def test_old_pca_state_format_is_reported(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """A ``pca_state.npz`` lacking an entry is reported as unreadable."""
    run_dir = _run(_fit_config(settings, spectra_paths), tmp_path / "run")
    with np.load(run_dir / PCA_STATE_FILE) as state:
        entries = {name: state[name] for name in state.files if name != "scaling_scales"}
    np.savez(run_dir / PCA_STATE_FILE, **entries)

    with pytest.raises(RunArtifactError, match="outdated or broken"):
        load_fit_artifacts(run_dir)


def test_register_rejects_samples_of_other_files(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Artifacts must list exactly the configured files."""
    config = _fit_config(settings, spectra_paths)
    run_dir = _run(config, tmp_path / "run")
    files = config["files"]
    assert isinstance(files, list)
    config["files"] = files[:-1]

    with pytest.raises(ValueError, match="configured files"):
        register_fit_result(config, run_dir)
