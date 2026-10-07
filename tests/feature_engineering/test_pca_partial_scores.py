"""Tests for partial score trajectories of flattened feature rows."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.pca import (
    ImputeStrategy,
    PcaModel,
    fit_pca,
    partial_scores,
    prepare_rows,
    time_point_order,
    transform_pca,
)
from flat_pca.feature_engineering.scaling import ScalingStrategy

# Columns out of (Step, Sequence, StepTime) order, two wavelengths per point.
FEATURES = (
    "400.0nm_2_1_0.00",
    "410.0nm_2_1_0.00",
    "400.0nm_1_2_0.00",
    "410.0nm_1_2_0.00",
    "410.0nm_1_1_1.00",
    "400.0nm_1_1_1.00",
    "400.0nm_1_1_0.50",
    "410.0nm_1_1_0.50",
)
# Time points in ascending order and the FEATURES positions of each.
POINTS = ((1, 1, 0.5), (1, 1, 1.0), (1, 2, 0.0), (2, 1, 0.0))
POINT_FEATURES = ((6, 7), (4, 5), (2, 3), (0, 1))


@pytest.fixture
def values() -> np.ndarray:
    """Return 24 correlated rows, two of which have a missing value.

    Returns
    -------
    np.ndarray
        Rows shaped ``(24, 8)``; rows 2 and 9 hold NaN.
    """
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(24, 3)) @ rng.normal(size=(3, len(FEATURES))) + 4.0
    matrix += 0.1 * rng.normal(size=matrix.shape)
    matrix[2, 1] = np.nan
    matrix[9, 6] = np.nan
    return matrix


def _fit(
    values: np.ndarray,
    impute_strategy: ImputeStrategy = "median",
    scaling_strategy: ScalingStrategy = "z-score",
) -> PcaModel:
    """Fit a three-component model of the rows.

    Parameters
    ----------
    values : np.ndarray
        Training rows.
    impute_strategy : ImputeStrategy, default ``"median"``
        Missing-value handling.
    scaling_strategy : ScalingStrategy, default ``"z-score"``
        Feature scaling.

    Returns
    -------
    PcaModel
        Fitted model without whitening.
    """
    return fit_pca(
        pl.DataFrame(values, schema=list(FEATURES), orient="row").lazy(),
        list(FEATURES),
        max_n_component=3,
        impute_strategy=impute_strategy,
        scaling_strategy=scaling_strategy,
    )


def _transformed(values: np.ndarray, model: PcaModel) -> np.ndarray:
    """Return the scores of ``transform_pca`` for the rows it keeps."""
    return (
        transform_pca(pl.DataFrame(values, schema=list(FEATURES), orient="row").lazy(), model)
        .select(model.pca_column_names)
        .collect()
        .to_numpy()
    )


@pytest.mark.parametrize("scaling_strategy", ["z-score", "none", "pareto"])
def test_end_points_match_the_scores_with_imputed_rows(
    values: np.ndarray, scaling_strategy: ScalingStrategy
) -> None:
    """The last point is the score, including rows whose values were imputed."""
    model = _fit(values, scaling_strategy=scaling_strategy)
    prepared = prepare_rows(values, model)

    result = partial_scores(prepared.values, model, (1, 3))

    assert result.scores.dtype == np.float64
    assert result.scores.shape == (24, len(POINTS), 2)
    np.testing.assert_allclose(
        result.scores[:, -1, :], _transformed(values, model)[:, [0, 2]], atol=1e-10
    )
    np.testing.assert_allclose(
        result.scores[[2, 9], -1, :], prepared.scores[[2, 9]][:, [0, 2]], atol=1e-10
    )


def test_end_points_match_the_scores_of_kept_rows_under_drop(values: np.ndarray) -> None:
    """Under ``"drop"``, the kept rows' last points are their scores."""
    model = _fit(values, impute_strategy="drop")
    prepared = prepare_rows(values, model)

    result = partial_scores(prepared.values, model, (2,))

    assert result.scores.shape == (22, len(POINTS), 1)
    np.testing.assert_allclose(
        result.scores[:, -1, 0], _transformed(values, model)[:, 1], atol=1e-10
    )


def test_points_follow_step_sequence_and_step_time(values: np.ndarray) -> None:
    """Each point adds the products of its own features in ascending order."""
    model = _fit(values)
    prepared = prepare_rows(values, model)

    result = partial_scores(prepared.values[:1], model, (1,))

    np.testing.assert_array_equal(result.steps, [point[0] for point in POINTS])
    np.testing.assert_array_equal(result.sequences, [point[1] for point in POINTS])
    np.testing.assert_array_equal(result.step_times, [point[2] for point in POINTS])
    scaled = (
        model.scaling_model.apply(
            pl.DataFrame(prepared.values[:1], schema=list(FEATURES), orient="row").lazy(),
            list(FEATURES),
        )
        .collect()
        .to_numpy()[0]
    )
    products = (scaled - model.pca.mean_) * model.pca.components_[0]
    expected = np.cumsum([products[list(features)].sum() for features in POINT_FEATURES])
    np.testing.assert_allclose(result.scores[0, :, 0], expected, atol=1e-12)


def test_rejects_missing_values(values: np.ndarray) -> None:
    """Rows must be imputed first."""
    model = _fit(values)

    with pytest.raises(ValueError, match="imputed"):
        partial_scores(values[2:3], model, (1,))


def test_rejects_components_out_of_range(values: np.ndarray) -> None:
    """Component numbers are one-based and at most ``n_component``."""
    model = _fit(values)
    prepared = prepare_rows(values[:1], model)

    for component in (0, 4):
        with pytest.raises(ValueError, match="components must be in 1..3"):
            partial_scores(prepared.values, model, (1, component))


def test_rejects_values_of_another_width(values: np.ndarray) -> None:
    """Reject rows whose width differs from the feature count."""
    model = _fit(values)

    with pytest.raises(ValueError, match="expected"):
        partial_scores(np.zeros((1, 3)), model, (1,))


def test_reused_order_gives_the_same_trajectories(values: np.ndarray) -> None:
    """Passing a precomputed order matches computing it, row by row or together."""
    model = _fit(values)
    prepared = prepare_rows(values, model)
    order = time_point_order(model.columns)

    together = partial_scores(prepared.values, model, (1, 2))
    one_row = partial_scores(prepared.values[5:6], model, (1, 2), order)

    np.testing.assert_array_equal(order.steps, together.steps)
    np.testing.assert_allclose(one_row.scores[0], together.scores[5], atol=1e-12)
