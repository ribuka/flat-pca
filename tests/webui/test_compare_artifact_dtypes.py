"""Tests for the float32/float64 artifact comparison script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def test_compare_run_refits_with_both_dtypes(
    settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """A run's config is refitted per dtype and the results nearly agree."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in spectra_paths],
        {"target_steps": [1, 2]},
        {
            "n_component": None,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "pareto",
        },
        {"cumulative_explained_variance": 2, "alpha": 0.01},
        {"cumulative_explained_variance": 2, "alpha": 0.01},
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    table = compare_run(run_dir, tmp_path / "comparison")

    assert table["quantity"].to_list() == ["scores", "t2", "q", "reconstruction"]
    assert max(table["relative_diff"]) < 1e-4
    assert np.load(tmp_path / "comparison" / "float32" / "X.npy").dtype == np.float32
    assert np.load(tmp_path / "comparison" / "float64" / "X.npy").dtype == np.float64
