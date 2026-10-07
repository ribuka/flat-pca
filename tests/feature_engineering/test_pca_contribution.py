"""Tests for the contribution of a single PCA component."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA

from flat_pca.feature_engineering.pca import (
    PcaModel,
    component_contribution,
    fit_pca,
    transform_pca,
)
from flat_pca.feature_engineering.scaling import ScalingStrategy

FEATURES = ("feature_a", "feature_b", "feature_c", "feature_d")


@pytest.fixture
def frame() -> pl.DataFrame:
    """Return a training frame of 25 correlated samples.

    Returns
    -------
    pl.DataFrame
        Frame with the ``FEATURES`` columns.
    """
    rng = np.random.default_rng(3)
    mixing = rng.normal(size=(len(FEATURES), len(FEATURES)))
    values = rng.normal(size=(25, len(FEATURES))) @ mixing + 5.0
    return pl.DataFrame(values, schema=list(FEATURES), orient="row")


def _fit(frame: pl.DataFrame, scaling_strategy: ScalingStrategy) -> PcaModel:
    """Fit a model with as many components as features.

    Parameters
    ----------
    frame : pl.DataFrame
        Training data.
    scaling_strategy : ScalingStrategy
        Scaling applied before PCA.

    Returns
    -------
    PcaModel
        Fitted model.
    """
    return fit_pca(
        frame.lazy(),
        list(FEATURES),
        max_n_component=None,
        scaling_strategy=scaling_strategy,
    )


def _score_matrix(frame: pl.DataFrame, model: PcaModel) -> np.ndarray:
    """Return the score matrix of ``frame`` under ``model``.

    Parameters
    ----------
    frame : pl.DataFrame
        Data holding the feature columns.
    model : PcaModel
        Fitted model.

    Returns
    -------
    np.ndarray
        Scores shaped ``(n_rows, n_component)``.
    """
    return (
        transform_pca(frame.lazy(), model)
        .select(model.pca_column_names)
        .collect()
        .to_numpy()
    )


def _total_with_mean(model: PcaModel, scores: np.ndarray) -> np.ndarray:
    """Return the sum of every component's contribution plus the mean.

    Parameters
    ----------
    model : PcaModel
        Fitted model.
    scores : np.ndarray
        Scores of the model.

    Returns
    -------
    np.ndarray
        Contributions summed over components, plus the PCA mean mapped back
        to the original scale.
    """
    total = sum(
        component_contribution(
            scores, model.pca, model.feature_arrays.scaling, component
        )
        for component in range(1, model.n_component + 1)
    )
    mean = model.feature_arrays.scaling.unscale(
        np.asarray(model.pca.mean_, dtype=float)[np.newaxis, :]
    )
    return total + mean


@pytest.mark.parametrize("scaling_strategy", ["none", "z-score", "robust", "pareto"])
def test_contributions_plus_mean_match_full_reconstruction(
    frame: pl.DataFrame, scaling_strategy: ScalingStrategy
) -> None:
    """Every contribution plus the mean adds up to ``reconstruct``."""
    model = _fit(frame, scaling_strategy)
    scores = _score_matrix(frame, model)
    reconstructed = model.reconstruct(
        pl.DataFrame(scores, schema=list(model.pca_column_names), orient="row")
    )

    np.testing.assert_allclose(
        _total_with_mean(model, scores), reconstructed.to_numpy(), atol=1e-10
    )


def test_whitened_contributions_match_inverse_transform(frame: pl.DataFrame) -> None:
    """Whitened scores are multiplied back by the component deviations."""
    model = _fit(frame, "none")
    whitened = PCA(n_components=model.n_component, whiten=True).fit(frame.to_numpy())
    model = replace(model, pca=whitened)
    scores = whitened.transform(frame.to_numpy())

    np.testing.assert_allclose(
        _total_with_mean(model, scores), whitened.inverse_transform(scores), atol=1e-10
    )


@pytest.mark.parametrize("scaling_strategy", ["z-score", "pareto"])
def test_contribution_is_the_scaled_component_term(
    frame: pl.DataFrame, scaling_strategy: ScalingStrategy
) -> None:
    """Return ``t_k w_k`` multiplied by the scale, without the center."""
    model = _fit(frame, scaling_strategy)
    scores = _score_matrix(frame, model)
    scales = np.array([model.scaling_model.scales[column] for column in FEATURES])

    result = component_contribution(
        scores, model.pca, model.feature_arrays.scaling, 2
    )

    expected = np.outer(scores[:, 1], model.pca.components_[1]) * scales
    np.testing.assert_allclose(result, expected)


def test_pareto_scale_is_the_square_root_of_the_deviation(frame: pl.DataFrame) -> None:
    """Pareto contributions are multiplied by the square root of the deviation."""
    model = _fit(frame, "pareto")
    scores = _score_matrix(frame, model)
    deviations = frame.select(pl.all().std()).to_numpy()[0]

    result = component_contribution(
        scores, model.pca, model.feature_arrays.scaling, 1
    )

    expected = np.outer(scores[:, 0], model.pca.components_[0]) * np.sqrt(deviations)
    np.testing.assert_allclose(result, expected)


def test_no_scaling_returns_the_component_term(frame: pl.DataFrame) -> None:
    """Without scaling the contribution is ``t_k w_k`` itself."""
    model = _fit(frame, "none")
    scores = _score_matrix(frame, model)

    result = component_contribution(
        scores, model.pca, model.feature_arrays.scaling, 3
    )

    np.testing.assert_allclose(
        result, np.outer(scores[:, 2], model.pca.components_[2])
    )


@pytest.mark.parametrize("component", [0, 5])
def test_rejects_out_of_range_component(
    frame: pl.DataFrame, component: int
) -> None:
    """Reject component numbers outside ``1..n_component``."""
    model = _fit(frame, "none")
    scores = _score_matrix(frame, model)

    with pytest.raises(ValueError, match="component must be between"):
        component_contribution(
            scores, model.pca, model.feature_arrays.scaling, component
        )
