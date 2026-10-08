"""Tests for the transform job, calling ``run_transform`` without a child process."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fit_runs import register_fit_run, register_transform_run
from spectra import SPECTRA_SHORT_FILE, write_spectra

from flat_pca.spectral.schema import SOURCE_COLUMN
from flat_pca.webui.database import Database
from flat_pca.webui.jobs.fit_run import (
    SAMPLES_FILE,
    SCORES_FILE,
    X_FILE,
    build_fit_config,
    run_fit,
    score_frame,
)
from flat_pca.webui.jobs.progress import read_progress
from flat_pca.webui.jobs.transform_run import (
    align_features,
    build_transform_config,
    run_transform,
    transform_target_intensity,
)
from flat_pca.webui.services.fit_artifacts import (
    load_display_artifacts,
    load_fit_artifacts,
)
from flat_pca.webui.services.run_dirs import RunDirs, fit_run_reference, run_dirs
from flat_pca.webui.services.runs import get_run, insert_run
from flat_pca.webui.services.transform_artifacts import register_transform_result
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import FIT_JOB

STATISTICS = {"cumulative_explained_variance": 2, "alpha": 0.01}


def _fit_run(
    database: Database,
    settings: Settings,
    paths: list[Path],
    impute_strategy: str = "median",
) -> dict[str, object]:
    """Register the fit run ``fit-1`` of the synthetic spectra and return its row."""
    register_fit_run(database, settings, paths, "fit-1", impute_strategy)
    run = get_run(database, "fit-1")
    assert run is not None
    return run


def _transform(
    settings: Settings,
    fit_run: dict[str, object],
    paths: list[Path],
    run_id: str = "tr-1",
) -> tuple[dict[str, object], Path]:
    """Run a transform job of ``paths`` and return its configuration and directory."""
    config = json.loads(
        json.dumps(
            build_transform_config(
                settings,
                [{"stem": path.stem, "path": str(path), "lot": "Z"} for path in paths],
                fit_run,
            )
        )
    )
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True)
    run_transform(config, run_dir)
    return config, run_dir


def test_build_transform_config_takes_the_fit_run_settings(
    database: Database, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The configuration names the fit run and copies its preprocessing and statistics."""
    fit_run = _fit_run(database, settings, spectra_paths)
    fit_config = json.loads(str(fit_run["config_json"]))

    target = spectra_paths[0]
    config = build_transform_config(
        settings,
        [
            {"stem": "a", "path": str(target), "lot": "Z"},
            {"stem": "gone", "path": "gone.parquet"},
        ],
        fit_run,
    )

    assert config["fit_run_id"] == "fit-1"
    assert config["fit_run_dir"] == str(fit_run["artifact_dir"])
    assert config["preprocess"] == fit_config["preprocess"]
    assert config["mahalanobis"] == fit_config["mahalanobis"]
    assert config["spe"] == fit_config["spe"]
    # Each file's size and modification time identify the transformed contents.
    assert config["files"] == [
        {
            "stem": "a",
            "path": str(target),
            "metadata": {"lot": "Z", "date": None, "yield_pct": None},
            "size": target.stat().st_size,
            "mtime_ns": target.stat().st_mtime_ns,
        },
        {
            "stem": "gone",
            "path": "gone.parquet",
            "metadata": {"lot": None, "date": None, "yield_pct": None},
            "size": None,
            "mtime_ns": None,
        },
    ]


@pytest.mark.parametrize("impute_strategy", ["drop", "median"])
def test_transform_of_the_fitted_files_reproduces_the_fit_scores(
    database: Database, settings: Settings, spectra_paths: list[Path], impute_strategy: str
) -> None:
    """Transforming the fitted files gives the fit run's matrix and its model's scores, T², and Q."""
    fit_run = _fit_run(database, settings, spectra_paths, impute_strategy)
    config, run_dir = _transform(settings, fit_run, spectra_paths)

    fit = load_fit_artifacts(Path(str(fit_run["artifact_dir"])))
    shown = load_display_artifacts(
        RunDirs(model=Path(str(fit_run["artifact_dir"])), data=run_dir)
    )
    np.testing.assert_allclose(shown.x, fit.x)
    assert shown.samples["stem"].to_list() == fit.samples["stem"].to_list()
    assert shown.samples["lot"].to_list() == ["Z"] * len(spectra_paths)
    fitted = pl.DataFrame(
        np.asarray(fit.x, dtype=np.float64), schema=fit.features["feature"].to_list()
    ).insert_column(0, fit.samples[SOURCE_COLUMN])
    expected = score_frame(fitted.lazy(), fit.model, json.loads(str(fit_run["config_json"])))
    assert shown.scores[SOURCE_COLUMN].to_list() == expected[SOURCE_COLUMN].to_list()
    assert shown.scores.columns == expected.columns
    np.testing.assert_allclose(
        shown.scores.select(pl.exclude(SOURCE_COLUMN)).to_numpy(),
        expected.select(pl.exclude(SOURCE_COLUMN)).to_numpy(),
        # The expected scores come from the float32 X.npy.
        rtol=1e-4,
        atol=1e-4,
    )
    progress = read_progress(run_dir)
    assert progress is not None
    assert (progress.stage, progress.done, progress.total) == ("save", 3, 3)
    assert register_transform_result(config, run_dir) == {
        "n_files": len(spectra_paths),
        "n_features": fit.features.height,
        "n_components": fit.model.n_component,
    }


def test_missing_features_are_imputed_and_extra_ones_dropped(
    database: Database, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Targets on other wavelengths keep the fit features; the missing ones are imputed."""
    fit_run = _fit_run(database, settings, spectra_paths)
    targets = write_spectra(
        tmp_path / "other", n_files=3, wavelengths=(400.0, 401.0, 420.0)
    )

    _, run_dir = _transform(settings, fit_run, targets)

    fit_dir = Path(str(fit_run["artifact_dir"]))
    shown = load_display_artifacts(RunDirs(model=fit_dir, data=run_dir))
    assert shown.x.shape == (3, shown.features.height)
    missing = ~shown.features["wavelength"].is_in([400.0, 401.0]).to_numpy()
    assert np.isnan(shown.x[:, missing]).all()
    assert not np.isnan(shown.x[:, ~missing]).any()
    # The median imputation scores every target.
    assert shown.scores.height == 3
    assert [Path(source).stem for source in shown.scores[SOURCE_COLUMN]] == [
        path.stem for path in targets
    ]


def test_targets_without_any_fit_feature_fail(
    database: Database, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Targets sharing no feature with the fit run cannot be transformed."""
    fit_run = _fit_run(database, settings, spectra_paths)
    targets = write_spectra(tmp_path / "other", n_files=2, wavelengths=(500.0, 501.0))

    with pytest.raises(ValueError, match="none of the fit run's features"):
        _transform(settings, fit_run, targets)


def test_drop_imputation_leaves_incomplete_targets_unscored(
    database: Database, settings: Settings, spectra_paths: list[Path]
) -> None:
    """With ``impute_strategy="drop"``, a target with missing values keeps its row but no score."""
    fit_run = _fit_run(database, settings, spectra_paths, impute_strategy="drop")
    targets = spectra_paths[SPECTRA_SHORT_FILE - 1 : SPECTRA_SHORT_FILE + 1]

    config, run_dir = _transform(settings, fit_run, targets)

    samples = pl.read_parquet(run_dir / SAMPLES_FILE)
    scores = pl.read_parquet(run_dir / SCORES_FILE)
    assert samples["stem"].to_list() == [path.stem for path in targets]
    assert np.load(run_dir / X_FILE).shape[0] == 2
    assert [Path(source).stem for source in scores[SOURCE_COLUMN]] == [targets[0].stem]
    assert register_transform_result(config, run_dir)["n_files"] == 2


def test_register_rejects_samples_of_other_files(
    database: Database, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Artifacts that do not list the configured files fail the run."""
    fit_run = _fit_run(database, settings, spectra_paths)
    config, run_dir = _transform(settings, fit_run, spectra_paths[:2])
    config["files"] = config["files"][:1]  # type: ignore[index]

    with pytest.raises(ValueError, match=SAMPLES_FILE):
        register_transform_result(config, run_dir)


def test_align_features_orders_the_fit_features() -> None:
    """The fit features come in their order, missing ones as nulls, extra ones dropped."""
    flattened = pl.DataFrame(
        {
            SOURCE_COLUMN: ["a"],
            "401.0nm_1_1_0.00": [2.0],
            "400.0nm_1_1_0.00": [1.0],
            "999.0nm_1_1_0.00": [9.0],
        }
    )

    aligned = align_features(
        flattened, ["400.0nm_1_1_0.00", "402.0nm_1_1_0.00", "401.0nm_1_1_0.00"]
    )

    assert aligned.columns == [
        SOURCE_COLUMN,
        "400.0nm_1_1_0.00",
        "402.0nm_1_1_0.00",
        "401.0nm_1_1_0.00",
    ]
    assert aligned.row(0) == ("a", 1.0, None, 2.0)


def test_run_dirs_of_fit_and_transform_runs(
    database: Database, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A fit run shows its own model and data; a transform run its fit run's model."""
    register_fit_run(database, settings, spectra_paths, "fit-1", "median")
    register_transform_run(database, settings, "fit-1", spectra_paths[:2], "tr-1")
    fit_run = get_run(database, "fit-1")
    transform_run = get_run(database, "tr-1")
    assert fit_run is not None
    assert transform_run is not None
    fit_dir = Path(str(fit_run["artifact_dir"]))

    assert fit_run_reference(fit_run) is None
    assert run_dirs(fit_run) == RunDirs(model=fit_dir, data=fit_dir)
    assert fit_run_reference(transform_run) == ("fit-1", fit_dir)
    assert run_dirs(transform_run) == RunDirs(
        model=fit_dir, data=Path(str(transform_run["artifact_dir"]))
    )
    assert transform_run["n_files"] == 2


def test_downsampled_fit_keeps_shared_grid_points_of_another_grid(
    database: Database, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Targets on another grid keep the values the downsampled fit run kept."""
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in spectra_paths],
        {"target_steps": [1, 2], "max_null_ratio": 0.1, "w_downsampling_stride": 2},
        {
            "n_component": 2,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "none",
        },
        STATISTICS,
        STATISTICS,
    )
    fit_dir = settings.runs_dir / "fit-1"
    fit_dir.mkdir(parents=True)
    run_fit(json.loads(json.dumps(config)), fit_dir)
    insert_run(database, "fit-1", FIT_JOB, config, fit_dir)
    fit_run = get_run(database, "fit-1")
    assert fit_run is not None
    targets = write_spectra(
        tmp_path / "other", n_files=2, wavelengths=(401.0, 402.5, 405.0, 410.0)
    )

    _, run_dir = _transform(settings, fit_run, targets)

    shown = load_display_artifacts(RunDirs(model=fit_dir, data=run_dir))
    wavelengths = shown.features["wavelength"].to_numpy()
    assert sorted(set(wavelengths.tolist())) == [400.0, 402.5, 410.0]
    assert np.isnan(shown.x[:, wavelengths == 400.0]).all()
    assert not np.isnan(shown.x[:, wavelengths != 400.0]).any()


def test_values_dropped_by_the_fit_downsampling_skip_the_log1p_check(
    database: Database, settings: Settings, tmp_path: Path
) -> None:
    """The fitted files transform again although ``log1p`` rejects a value the fit dropped."""
    paths = write_spectra(tmp_path / "negative")
    for path in paths:
        pl.read_parquet(path).with_columns(pl.lit(-2.0).alias("401.0nm")).write_parquet(
            path
        )
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in paths],
        {
            "target_steps": [1, 2],
            "max_null_ratio": 0.1,
            "w_downsampling_stride": 2,
            "intensity_transform": "log1p",
        },
        {
            "n_component": 2,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "none",
        },
        STATISTICS,
        STATISTICS,
    )
    fit_dir = settings.runs_dir / "fit-1"
    fit_dir.mkdir(parents=True)
    run_fit(json.loads(json.dumps(config)), fit_dir)
    insert_run(database, "fit-1", FIT_JOB, config, fit_dir)
    fit_run = get_run(database, "fit-1")
    assert fit_run is not None

    _, run_dir = _transform(settings, fit_run, paths)

    fit = load_fit_artifacts(fit_dir)
    np.testing.assert_allclose(np.load(run_dir / X_FILE), fit.x, rtol=1e-6)


def test_intensity_transform_keeps_rows_of_a_square_matrix() -> None:
    """As many files as features keep each file's values in its own row."""
    aligned = pl.DataFrame(
        {
            SOURCE_COLUMN: ["a", "b"],
            "400.0nm_1_1_0.00": [4.0, 9.0],
            "401.0nm_1_1_0.00": [16.0, None],
        }
    )

    transformed = transform_target_intensity(
        aligned,
        ["400.0nm_1_1_0.00", "401.0nm_1_1_0.00"],
        {"intensity_transform": "sqrt"},
    )

    assert transformed.columns == aligned.columns
    assert transformed.row(0) == ("a", 2.0, 4.0)
    assert transformed.row(1)[:2] == ("b", 3.0)
    assert np.isnan(transformed.row(1)[2])
