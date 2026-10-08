"""Tests for the float32/float64 artifact comparison script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from preprocess_settings import preprocess_settings
from spectra import write_spectra

from flat_pca.webui.compare_artifact_dtypes import compare_run, compare_values
from flat_pca.webui.jobs.fit_run import build_fit_config
from flat_pca.webui.settings import Settings


def test_compare_values_aligns_score_signs() -> None:
    """Flipped component signs do not count as differences."""
    double = {"scores": np.array([[1.0, -2.0], [3.0, 4.0]]), "q": np.array([1.0, 2.0])}
    single = {"scores": np.array([[-1.0, -2.0], [-3.0, 4.5]]), "q": np.array([1.0, 2.5])}

    table = compare_values(single, double)

    assert table["quantity"].to_list() == ["scores", "q"]
    assert table["max_abs_diff"].to_list() == [0.5, 0.5]
    assert table["relative_diff"].to_list() == [0.125, 0.25]


def _write_config(
    settings: Settings, paths: list[Path], run_dir: Path, **pca: object
) -> None:
    """Write the ``config.json`` of a fit run over ``paths``."""
    run_dir.mkdir()
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in paths],
        preprocess_settings(),
        {
            "n_component": None,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "pareto",
            **pca,
        },
        {"cumulative_explained_variance": 2, "alpha": 0.01},
        {"cumulative_explained_variance": 2, "alpha": 0.01},
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_compare_run_saves_one_fit_in_both_dtypes(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The float32 copy holds the float64 fit's matrices rounded to float32."""
    _write_config(settings, spectra_paths, tmp_path / "run")

    table = compare_run(tmp_path / "run", tmp_path / "comparison")

    assert table["quantity"].to_list() == ["scores", "t2", "q", "reconstruction"]
    assert max(table["relative_diff"]) < 1e-5
    single = tmp_path / "comparison" / "float32"
    double = tmp_path / "comparison" / "float64"
    for name in ("X.npy", "components.npy"):
        assert np.load(single / name).dtype == np.float32
        np.testing.assert_array_equal(
            np.load(single / name), np.load(double / name).astype(np.float32)
        )


def test_compare_run_excludes_randomized_solver_differences(
    settings: Settings, tmp_path: Path
) -> None:
    """Data fitted by the randomized PCA solver differ only by float32 rounding."""
    paths = write_spectra(
        tmp_path / "large", n_files=60, wavelengths=[400.0 + 0.5 * i for i in range(86)]
    )
    _write_config(
        settings,
        paths,
        tmp_path / "run",
        n_component=10,
        impute_strategy="drop",
        scaling_strategy="none",
    )

    table = compare_run(tmp_path / "run", tmp_path / "comparison")

    assert max(table["relative_diff"]) < 1e-5
