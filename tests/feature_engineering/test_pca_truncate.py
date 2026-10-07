"""Tests for truncating a fitted PCA pipeline to its leading components."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA

from flat_pca.feature_engineering.pca import PcaModel, fit_pca, truncate_pca_model

COLUMNS = [f"f{index}" for index in range(8)]


@pytest.fixture
def frame() -> pl.LazyFrame:
    """Return random features with more samples than columns."""
    rng = np.random.default_rng(5)
    return pl.DataFrame(rng.normal(size=(30, len(COLUMNS))), schema=COLUMNS).lazy()


@pytest.fixture
def model(frame: pl.LazyFrame) -> PcaModel:
    """Return a model fitted with every component."""
    return fit_pca(
        frame, COLUMNS, n_component=None, max_n_component=None, scaling_strategy="none"
    )


def test_truncation_matches_a_fit_with_fewer_components(
    frame: pl.LazyFrame, model: PcaModel
) -> None:
    """Kept components and the noise variance match a direct three-component fit."""
    truncated = truncate_pca_model(model, 3)
    direct = PCA(n_components=3).fit(frame.collect().to_numpy())

    assert truncated.n_component == 3
    assert truncated.pca_column_names == ("pca-1", "pca-2", "pca-3")
    np.testing.assert_allclose(
        np.abs(truncated.pca.components_), np.abs(direct.components_), atol=1e-10
    )
    np.testing.assert_allclose(
        truncated.pca.explained_variance_, direct.explained_variance_
    )
    assert truncated.pca.noise_variance_ == pytest.approx(direct.noise_variance_)
    assert truncated.pca.n_components_ == 3


def test_truncation_keeps_the_total_residual_variance(model: PcaModel) -> None:
    """Folding the dropped variances into the noise variance keeps theta_1."""
    truncated = truncate_pca_model(model, 2)

    dropped = np.sum(model.pca.explained_variance_[2:])
    folded = truncated.pca.noise_variance_ * (len(COLUMNS) - 2)
    assert folded == pytest.approx(dropped)
    assert truncated.get_spe_threshold(0.01, 2) != pytest.approx(
        model.get_spe_threshold(0.01, 2)
    )


def test_truncation_to_the_fitted_count_returns_the_model(model: PcaModel) -> None:
    """Keeping every component returns the model unchanged."""
    assert truncate_pca_model(model, model.n_component) is model


@pytest.mark.parametrize("n_component", [0, 9, True])
def test_truncation_rejects_counts_outside_the_fitted_range(
    model: PcaModel, n_component: int
) -> None:
    """Counts outside ``1..`` the fitted count raise ``ValueError``."""
    with pytest.raises(ValueError, match="n_component must be between"):
        truncate_pca_model(model, n_component)
