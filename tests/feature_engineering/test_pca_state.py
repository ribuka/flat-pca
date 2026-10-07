"""Tests for saving a fitted PCA pipeline as NumPy arrays and restoring it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.pca import PcaModel, SpeConfig, fit_pca, transform_pca

COLUMNS = [f"f{index}" for index in range(6)]


@pytest.fixture
def frame() -> pl.LazyFrame:
    """Return correlated features with two missing values."""
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(40, 2))
    values = latent @ rng.normal(size=(2, len(COLUMNS))) + 0.1 * rng.normal(
        size=(40, len(COLUMNS))
    )
    values[2, 1] = np.nan
    values[9, 4] = np.nan
    return pl.DataFrame(values, schema=COLUMNS).lazy()


def _scores(frame: pl.LazyFrame, model: PcaModel) -> np.ndarray:
    """Return the scores and Q of ``frame`` under ``model``."""
    return (
        transform_pca(frame, model, spe=SpeConfig(cumulative_explained_variance=2))
        .select(*model.pca_column_names, "spe", "spe_ucl")
        .collect()
        .to_numpy()
    )


def _save_and_load(model: PcaModel, directory: Path, dtype: str) -> PcaModel:
    """Save ``model`` to ``.npy``/``.npz`` files and restore it."""
    np.save(directory / "components.npy", model.pca.components_.astype(dtype))
    np.savez(directory / "pca_state.npz", **model.to_pca_state())
    components = np.load(directory / "components.npy", mmap_mode="r")
    with np.load(directory / "pca_state.npz") as state:
        return PcaModel.from_pca_state(COLUMNS, components, state)


@pytest.mark.parametrize("impute_strategy", ["drop", "median", "kmeans"])
@pytest.mark.parametrize("scaling_strategy", ["none", "robust", "pareto"])
@pytest.mark.parametrize("outlier_strategy", [None, "winsorize"])
def test_saved_state_restores_the_transform(
    frame: pl.LazyFrame,
    tmp_path: Path,
    impute_strategy: str,
    scaling_strategy: str,
    outlier_strategy: str | None,
) -> None:
    """A model saved in float64 transforms exactly like the fitted one."""
    model = fit_pca(
        frame,
        COLUMNS,
        n_component=3,
        impute_strategy=impute_strategy,  # type: ignore[arg-type]
        outlier_strategy=outlier_strategy,  # type: ignore[arg-type]
        scaling_strategy=scaling_strategy,  # type: ignore[arg-type]
    )

    restored = _save_and_load(model, tmp_path, "float64")

    assert restored.columns == model.columns
    assert restored.impute_model == model.impute_model
    assert restored.outlier_model == model.outlier_model
    assert restored.scaling_model == model.scaling_model
    assert restored.pca_column_names == model.pca_column_names
    np.testing.assert_allclose(_scores(frame, restored), _scores(frame, model), rtol=1e-12)


def test_float32_components_are_computed_in_float64(
    frame: pl.LazyFrame, tmp_path: Path
) -> None:
    """float32 components are converted to float64 and change scores slightly."""
    model = fit_pca(frame, COLUMNS, n_component=3, scaling_strategy="pareto")

    restored = _save_and_load(model, tmp_path, "float32")

    assert restored.pca.components_.dtype == np.float64
    np.testing.assert_allclose(
        _scores(frame, restored), _scores(frame, model), rtol=1e-5, atol=1e-6
    )


def test_stages_without_values_store_empty_arrays(frame: pl.LazyFrame) -> None:
    """Strategies without fitted values save empty per-feature arrays."""
    model = fit_pca(frame, COLUMNS, n_component=2, scaling_strategy="none")

    state = model.to_pca_state()

    assert str(state["impute_strategy"]) == "drop"
    assert state["impute_values"].shape == (0,)
    assert state["impute_kmeans_centroids"].shape == (0, len(COLUMNS))
    assert str(state["outlier_strategy"]) == ""
    assert state["scaling_centers"].shape == (0,)
    assert "components" not in state


def test_kmeans_state_orders_centroids_by_column(frame: pl.LazyFrame) -> None:
    """kmeans centroids are saved as an (n_clusters, features) matrix."""
    model = fit_pca(
        frame,
        COLUMNS,
        n_component=2,
        impute_strategy="kmeans",
        impute_kmeans_n_clusters=3,
    )

    state = model.to_pca_state()

    assert int(state["impute_kmeans_n_clusters"]) == 3
    expected = [
        [centroid[column] for column in COLUMNS]
        for centroid in model.impute_model.kmeans_centroids
    ]
    np.testing.assert_array_equal(state["impute_kmeans_centroids"], expected)


def test_mismatched_components_are_rejected(frame: pl.LazyFrame) -> None:
    """Components that do not fit the columns raise ``ValueError``."""
    model = fit_pca(frame, COLUMNS, n_component=2)

    with pytest.raises(ValueError, match="components must be shaped"):
        PcaModel.from_pca_state(
            COLUMNS[:-1], model.pca.components_, model.to_pca_state()
        )


def test_missing_state_entry_raises_key_error(frame: pl.LazyFrame) -> None:
    """State written without an entry, e.g. in an older format, is rejected."""
    model = fit_pca(frame, COLUMNS, n_component=2)
    state = model.to_pca_state()
    del state["scaling_scales"]

    with pytest.raises(KeyError):
        PcaModel.from_pca_state(COLUMNS, model.pca.components_, state)
