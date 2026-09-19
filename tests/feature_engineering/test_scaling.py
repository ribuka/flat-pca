"""Tests for feature scaling strategies."""

import polars as pl

from flat_pca.feature_engineering.scaling import apply_scaler, fit_scaler


def test_none_scaling_preserves_values_and_dtypes() -> None:
    """Leave selected columns unchanged when scaling is disabled."""
    frame = pl.DataFrame(
        {
            "filename": ["a", "b", "c"],
            "feature_int": [1, 2, 3],
            "feature_float": [10.0, 20.0, 30.0],
        }
    )
    columns = ["feature_int", "feature_float"]

    model = fit_scaler(frame.lazy(), columns, "none")
    result = apply_scaler(frame.lazy(), columns, model).collect()

    assert model.centers == {"feature_int": 0.0, "feature_float": 0.0}
    assert model.scales == {"feature_int": 1.0, "feature_float": 1.0}
    assert result.equals(frame)
