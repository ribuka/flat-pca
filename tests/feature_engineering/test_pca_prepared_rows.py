"""Tests for preparing feature rows with a fitted model."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.pca import (
    ImputeStrategy,
    PcaModel,
    fit_pca,
    prepare_rows,
    transform_pca,
)
from flat_pca.feature_engineering.pca.reconstruct import reconstruct_standardized

FEATURES = ("feature_a", "feature_b", "feature_c")


@pytest.fixture
def values() -> np.ndarray:
    """Return 20 correlated rows, two of which have a missing value.

    Returns
    -------
    np.ndarray
        Rows shaped ``(20, 3)``; rows 3 and 11 hold NaN.
    """
    rng = np.random.default_rng(7)
    mixing = rng.normal(size=(len(FEATURES), len(FEATURES)))
    matrix = rng.normal(size=(20, len(FEATURES))) @ mixing + 2.0
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


@pytest.mark.parametrize("impute_strategy", ["median", "kmeans"])
def test_imputed_rows_have_no_missing_value(
    values: np.ndarray, impute_strategy: ImputeStrategy
) -> None:
    """Keep every row and fill the missing values."""
    model = _fit(values, impute_strategy)

    prepared = prepare_rows(values, model)

    np.testing.assert_array_equal(prepared.kept, np.arange(20))
    assert np.isfinite(prepared.values).all()
    np.testing.assert_array_equal(prepared.values[0], values[0])


@pytest.mark.parametrize("impute_strategy", ["drop", "median", "kmeans"])
def test_scores_match_transform_pca(
    values: np.ndarray, impute_strategy: ImputeStrategy
) -> None:
    """The scores agree with ``transform_pca`` of the same rows."""
    model = _fit(values, impute_strategy)
    expected = (
        transform_pca(
            pl.DataFrame(values, schema=list(FEATURES), orient="row").lazy(), model
        )
        .select(model.pca_column_names)
        .collect()
        .to_numpy()
    )

    prepared = prepare_rows(values, model)

    np.testing.assert_allclose(prepared.scores, expected)


def test_drop_strategy_drops_rows_with_a_missing_value(values: np.ndarray) -> None:
    """Rows with a missing value are not kept under ``"drop"``."""
    model = _fit(values, "drop")

    prepared = prepare_rows(values[[0, 3, 4, 11]], model)

    np.testing.assert_array_equal(prepared.kept, [0, 2])
    np.testing.assert_array_equal(prepared.values, values[[0, 4]])
    assert prepared.scores.shape == (2, model.n_component)


def test_drop_strategy_without_complete_rows_returns_empty(values: np.ndarray) -> None:
    """Return empty arrays when every row is dropped."""
    model = _fit(values, "drop")

    prepared = prepare_rows(values[[3, 11]], model)

    assert prepared.kept.size == 0
    assert prepared.values.shape == (0, len(FEATURES))
    assert prepared.scores.shape == (0, model.n_component)


def test_residual_of_imputed_rows_has_no_nan(values: np.ndarray) -> None:
    """Residuals of imputed rows are finite and vanish with every component."""
    model = _fit(values, "median")
    prepared = prepare_rows(values, model)

    scaling = model.feature_arrays.scaling
    leading = scaling.unscale(reconstruct_standardized(prepared.scores, model.pca, 1))
    full = scaling.unscale(
        reconstruct_standardized(prepared.scores, model.pca, model.n_component)
    )

    assert np.isfinite(prepared.values - leading).all()
    np.testing.assert_allclose(prepared.values - full, 0.0, atol=1e-10)


def test_rejects_values_of_another_width(values: np.ndarray) -> None:
    """Reject rows whose width differs from the feature count."""
    model = _fit(values, "median")

    with pytest.raises(ValueError, match="expected"):
        prepare_rows(values[:, :2], model)
