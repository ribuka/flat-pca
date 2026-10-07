"""Tests for the preprocessing parameters shown on the model screen."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.outlier import OutlierStrategy
from flat_pca.feature_engineering.pca import ImputeStrategy, PcaModel, fit_pca
from flat_pca.feature_engineering.scaling import ScalingStrategy
from flat_pca.webui.services.preprocess_parameters import (
    is_parameter_key,
    parameter_options,
    parameter_values,
)

FEATURES = ["feature_a", "feature_b", "feature_c"]


def _fit(
    impute_strategy: ImputeStrategy,
    outlier_strategy: OutlierStrategy,
    scaling_strategy: ScalingStrategy,
) -> PcaModel:
    """Fit a two-component model on 30 rows, one of them with a missing value."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(30, len(FEATURES))) * [1.0, 2.0, 3.0] + [1.0, 5.0, -2.0]
    matrix[4, 1] = np.nan
    return fit_pca(
        pl.DataFrame(matrix, schema=FEATURES).lazy(),
        FEATURES,
        n_component=2,
        impute_strategy=impute_strategy,
        outlier_strategy=outlier_strategy,
        scaling_strategy=scaling_strategy,
        impute_kmeans_n_clusters=2 if impute_strategy == "kmeans" else None,
    )


def test_minimal_pipeline_holds_only_the_centering_mean() -> None:
    """Without scaling, imputation values, or outlier handling, only the mean is offered."""
    model = _fit("drop", None, "none")

    assert parameter_options(model) == {"mean": "中心化の平均"}
    np.testing.assert_array_equal(parameter_values(model, "mean"), model.pca.mean_)
    with pytest.raises(ValueError, match="scaling_scale"):
        parameter_values(model, "scaling_scale")


def test_full_pipeline_offers_every_parameter_in_column_order() -> None:
    """Scaling, median imputation, and outlier handling add their per-feature values."""
    model = _fit("median", "winsorize", "z-score")

    assert list(parameter_options(model)) == [
        "mean",
        "scaling_center",
        "scaling_scale",
        "impute_median",
        "outlier_lower",
        "outlier_upper",
        "winsor_lower",
        "winsor_upper",
    ]
    expected = {
        "scaling_center": model.scaling_model.centers,
        "scaling_scale": model.scaling_model.scales,
        "impute_median": model.impute_model.values,
        "outlier_lower": model.outlier_model.bounds.outlier_lower,
        "outlier_upper": model.outlier_model.bounds.outlier_upper,
        "winsor_lower": model.outlier_model.bounds.winsor_lower,
        "winsor_upper": model.outlier_model.bounds.winsor_upper,
    }
    for key, by_column in expected.items():
        np.testing.assert_allclose(
            parameter_values(model, key), [by_column[column] for column in FEATURES]
        )


def test_kmeans_imputation_offers_one_centroid_per_cluster() -> None:
    """Each kmeans centroid is a separate parameter."""
    model = _fit("kmeans", None, "none")

    options = parameter_options(model)

    assert [key for key in options if key.startswith("kmeans")] == [
        "kmeans_centroid_1",
        "kmeans_centroid_2",
    ]
    assert options["kmeans_centroid_2"] == "補完値（kmeans 重心 2）"
    centroid = model.impute_model.kmeans_centroids[1]
    np.testing.assert_allclose(
        parameter_values(model, "kmeans_centroid_2"),
        [centroid[column] for column in FEATURES],
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("mean", True),
        ("winsor_upper", True),
        ("kmeans_centroid_3", True),
        ("kmeans_centroid_0", False),
        ("kmeans_centroid_", False),
        ("component", False),
        ("median", False),
    ],
)
def test_parameter_keys_are_recognized(key: str, expected: bool) -> None:
    """Fixed keys and 1-based centroid keys are parameter keys."""
    assert is_parameter_key(key) is expected
