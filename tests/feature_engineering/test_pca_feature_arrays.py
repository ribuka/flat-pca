"""Tests that the array form of the fitted preprocessing agrees with Polars."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.outlier import OutlierStrategy
from flat_pca.feature_engineering.pca import (
    ImputeStrategy,
    PcaModel,
    fit_pca,
    prepare_rows,
)
from flat_pca.feature_engineering.pca.prepared_rows import scale_rows
from flat_pca.feature_engineering.scaling import ScalingStrategy

FEATURES = ("feature_a", "feature_b", "feature_c", "feature_d")


@pytest.fixture
def training() -> np.ndarray:
    """Return 40 correlated training rows with missing values and outliers.

    Returns
    -------
    np.ndarray
        Rows shaped ``(40, 4)``; rows 5 and 17 hold NaN, and rows 2 and 9
        hold an extreme value.
    """
    rng = np.random.default_rng(11)
    mixing = rng.normal(size=(len(FEATURES), len(FEATURES)))
    matrix = rng.normal(size=(40, len(FEATURES))) @ mixing + 3.0
    matrix[5, 1] = np.nan
    matrix[17, 3] = np.nan
    matrix[2, 0] = 40.0
    matrix[9, 2] = -35.0
    return matrix


@pytest.fixture
def rows() -> np.ndarray:
    """Return new rows mixing complete, missing, and outlying values.

    Returns
    -------
    np.ndarray
        Rows shaped ``(8, 4)``: row 1 holds NaN, row 3 an extreme value,
        row 6 both, and the others are ordinary.
    """
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(8, len(FEATURES))) * 2.0 + 3.0
    matrix[1, 2] = np.nan
    matrix[3, 1] = 60.0
    matrix[6, 0] = np.nan
    matrix[6, 3] = -50.0
    return matrix


def _fit(
    training: np.ndarray,
    impute_strategy: ImputeStrategy,
    outlier_strategy: OutlierStrategy,
    scaling_strategy: ScalingStrategy,
) -> PcaModel:
    """Fit a model with the given preprocessing.

    Parameters
    ----------
    training : np.ndarray
        Training rows.
    impute_strategy : ImputeStrategy
        Missing-value handling.
    outlier_strategy : OutlierStrategy
        Outlier handling.
    scaling_strategy : ScalingStrategy
        Feature scaling.

    Returns
    -------
    PcaModel
        Fitted model with three components.
    """
    return fit_pca(
        pl.DataFrame(training, schema=list(FEATURES), orient="row").lazy(),
        list(FEATURES),
        n_component=3,
        impute_strategy=impute_strategy,
        outlier_strategy=outlier_strategy,
        scaling_strategy=scaling_strategy,
        impute_kmeans_n_clusters=3 if impute_strategy == "kmeans" else None,
    )


def _prepare_with_polars(
    values: np.ndarray, model: PcaModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare rows through the Polars ``apply`` of each fitted stage.

    Parameters
    ----------
    values : np.ndarray
        Feature rows in the model's column order.
    model : PcaModel
        Fitted model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Kept row indices, prepared (unscaled) values, and scaled values.
    """
    columns = list(model.columns)
    frame = pl.from_numpy(values, schema=columns, orient="row").with_row_index("__row")
    prepared = model.outlier_model.apply(
        model.impute_model.apply(frame.lazy(), columns), columns
    ).collect()
    scaled = model.scaling_model.apply(prepared.lazy(), columns).select(columns)
    return (
        prepared["__row"].to_numpy().astype(np.intp),
        prepared.select(columns).to_numpy().astype(np.float64),
        scaled.collect().to_numpy().astype(np.float64),
    )


@pytest.mark.parametrize("impute_strategy", ["drop", "median", "kmeans"])
@pytest.mark.parametrize("outlier_strategy", [None, "winsorize", "drop"])
@pytest.mark.parametrize(
    "scaling_strategy", ["none", "z-score", "minmax", "robust", "pareto"]
)
def test_prepare_rows_matches_polars_stages(
    training: np.ndarray,
    rows: np.ndarray,
    impute_strategy: ImputeStrategy,
    outlier_strategy: OutlierStrategy,
    scaling_strategy: ScalingStrategy,
) -> None:
    """Kept rows, values, scaled values, and scores agree with Polars."""
    model = _fit(training, impute_strategy, outlier_strategy, scaling_strategy)
    kept, values, scaled = _prepare_with_polars(rows, model)

    prepared = prepare_rows(rows, model)

    np.testing.assert_array_equal(prepared.kept, kept)
    np.testing.assert_allclose(prepared.values, values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        scale_rows(prepared.values, model), scaled, rtol=1e-12, atol=1e-12
    )
    if kept.size:
        np.testing.assert_allclose(
            prepared.scores, model.pca.transform(scaled), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("outlier_strategy", ["winsorize", "drop"])
def test_outlier_rows_are_handled(
    training: np.ndarray, rows: np.ndarray, outlier_strategy: OutlierStrategy
) -> None:
    """The fixture rows exercise the outlier handling."""
    model = _fit(training, "median", outlier_strategy, "z-score")

    prepared = prepare_rows(rows, model)

    if outlier_strategy == "drop":
        assert 3 not in prepared.kept
    else:
        assert prepared.values[3, 1] < 60.0


def test_feature_arrays_are_built_once(training: np.ndarray) -> None:
    """The model keeps the arrays it built."""
    model = _fit(training, "median", None, "z-score")

    assert model.feature_arrays is model.feature_arrays
