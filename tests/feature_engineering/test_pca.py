"""Tests for PCA feature engineering."""

import numpy as np
import polars as pl

from spca.feature_engineering.pca import fit_pca, transform_pca


def test_none_scaling_leaves_centering_to_sklearn_pca() -> None:
    """Fit PCA on unscaled values and preserve source feature columns."""
    frame = pl.DataFrame(
        {
            "filename": ["a", "b", "c"],
            "feature_a": [1.0, 2.0, 4.0],
            "feature_b": [10.0, 30.0, 90.0],
        }
    )
    columns = ["feature_a", "feature_b"]

    model = fit_pca(
        frame.lazy(),
        columns,
        n_component=1,
        max_n_component=None,
        scaling_strategy="none",
    )
    result = transform_pca(frame.lazy(), model).collect()

    np.testing.assert_allclose(model.pca.mean_, [7.0 / 3.0, 130.0 / 3.0])
    assert model.scaling_model.strategy == "none"
    assert result.select(columns).equals(frame.select(columns))
    assert "pca-1" in result.columns
