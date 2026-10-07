"""Tests for the per-feature contributions to the Q statistic."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.pca import (
    ImputeStrategy,
    PcaModel,
    SpeConfig,
    fit_pca,
    prepare_rows,
    q_contribution,
    transform_pca,
)

FEATURES = ("feature_a", "feature_b", "feature_c", "feature_d")


@pytest.fixture
def values() -> np.ndarray:
    """Return 20 correlated rows, two of which have a missing value.

    Returns
    -------
    np.ndarray
        Rows shaped ``(20, 4)``; rows 3 and 11 hold NaN.
    """
    rng = np.random.default_rng(5)
    mixing = rng.normal(size=(len(FEATURES), len(FEATURES)))
    matrix = rng.normal(size=(20, len(FEATURES))) @ mixing + 3.0
    matrix[3, 0] = np.nan
    matrix[11, 2] = np.nan
    return matrix


def _fit(values: np.ndarray, impute_strategy: ImputeStrategy) -> PcaModel:
    """Fit a z-score model with as many components as features.

    Parameters
    ----------
    values : np.ndarray
        Training rows.
    impute_strategy : ImputeStrategy
        Missing-value handling.

    Returns
    -------
    PcaModel
        Fitted model.
    """
    return fit_pca(
        pl.DataFrame(values, schema=list(FEATURES), orient="row").lazy(),
        list(FEATURES),
        max_n_component=None,
        impute_strategy=impute_strategy,
        impute_kmeans_n_clusters=2 if impute_strategy == "kmeans" else None,
        scaling_strategy="z-score",
    )


def _spe(values: np.ndarray, model: PcaModel, config: SpeConfig) -> np.ndarray:
    """Return the Q column of ``transform_pca``."""
    return (
        transform_pca(
            pl.DataFrame(values, schema=list(FEATURES), orient="row").lazy(),
            model,
            spe=config,
        )
        .select(config.spe_column)
        .collect()
        .to_series()
        .to_numpy()
    )


@pytest.mark.parametrize("impute_strategy", ["median", "kmeans"])
@pytest.mark.parametrize("selector", [1, 2, 0.6])
def test_sum_matches_q_of_imputed_rows(
    values: np.ndarray, impute_strategy: ImputeStrategy, selector: float
) -> None:
    """Each row's contributions add up to its Q, imputed rows included."""
    model = _fit(values, impute_strategy)
    config = SpeConfig(cumulative_explained_variance=selector)

    contributions = q_contribution(prepare_rows(values, model), model, selector)

    assert contributions.shape == values.shape
    assert (contributions >= 0).all()
    np.testing.assert_allclose(contributions.sum(axis=1), _spe(values, model, config))


def test_dropped_rows_have_no_contribution(values: np.ndarray) -> None:
    """Under ``"drop"`` only the kept rows have contributions."""
    model = _fit(values, "drop")

    contributions = q_contribution(prepare_rows(values[[0, 3, 4]], model), model, 2)

    assert contributions.shape == (2, len(FEATURES))
    np.testing.assert_allclose(
        contributions.sum(axis=1),
        _spe(values[[0, 4]], model, SpeConfig(cumulative_explained_variance=2)),
    )


def test_every_component_leaves_no_residual(values: np.ndarray) -> None:
    """Reconstructing from every component leaves zero contributions."""
    model = _fit(values, "median")

    contributions = q_contribution(prepare_rows(values, model), model, None)

    np.testing.assert_allclose(contributions, 0.0, atol=1e-12)
