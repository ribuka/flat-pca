"""Tests for feature scaling strategies."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.scaling import (
    ScalingModel,
    apply_scaler,
    fit_scaler,
    pareto_scale,
)


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


def test_pareto_scaling_divides_centered_values_by_sqrt_of_std() -> None:
    """Center by the mean and divide by the square root of the std."""
    frame = pl.DataFrame(
        {
            "filename": ["a", "b", "c", "d"],
            "peak": [100.0, 400.0, 900.0, 1600.0],
            "dark": [1.0, 2.0, 1.0, 2.0],
        }
    )
    columns = ["peak", "dark"]
    values = frame.select(columns).to_numpy()
    expected_centers = values.mean(axis=0)
    expected_scales = np.sqrt(values.std(axis=0, ddof=1))

    model = fit_scaler(frame.lazy(), columns, "pareto")
    result = apply_scaler(frame.lazy(), columns, model).collect()

    assert model.strategy == "pareto"
    np.testing.assert_allclose(list(model.centers.values()), expected_centers)
    np.testing.assert_allclose(list(model.scales.values()), expected_scales)
    np.testing.assert_allclose(
        result.select(columns).to_numpy(),
        (values - expected_centers) / expected_scales,
    )
    assert result["filename"].equals(frame["filename"])


def test_pareto_scaling_compresses_variance_ratio_to_std_ratio() -> None:
    """Shrink a 10000:1 variance ratio to the 100:1 standard-deviation ratio.

    Pareto scaling leaves each column with a variance equal to its original
    standard deviation, between unscaled data (10000:1) and z-score (1:1).
    """
    frame = pl.DataFrame(
        {"peak": [0.0, 100.0, 200.0, 300.0], "dark": [0.0, 1.0, 2.0, 3.0]}
    )

    scaled = pareto_scale(frame.lazy(), ["peak", "dark"]).collect()

    variances = scaled.select(pl.all().var()).row(0)
    np.testing.assert_allclose(variances[0] / variances[1], 100.0)


def test_pareto_scaling_uses_unit_scale_for_constant_column() -> None:
    """Fall back to a unit scale when a column has zero variance."""
    frame = pl.DataFrame({"constant": [5.0, 5.0, 5.0], "varying": [1.0, 2.0, 3.0]})

    model = fit_scaler(frame.lazy(), ["constant", "varying"], "pareto")
    result = pareto_scale(frame.lazy(), ["constant", "varying"]).collect()

    assert model.scales["constant"] == 1.0
    assert result["constant"].to_list() == [0.0, 0.0, 0.0]


def test_scaling_model_apply_matches_apply_scaler() -> None:
    """Apply the same transformation as ``apply_scaler``."""
    frame = pl.DataFrame({"a": [1.0, 2.0, 4.0], "b": [10.0, 30.0, 90.0]})
    model = fit_scaler(frame.lazy(), ["a", "b"], "robust")

    result = model.apply(frame.lazy(), ["a", "b"]).collect()

    assert result.equals(apply_scaler(frame.lazy(), ["a", "b"], model).collect())


def test_scaling_model_payload_round_trip() -> None:
    """Restore the same model from its payload."""
    model = ScalingModel(strategy="pareto", centers={"a": 1.5}, scales={"a": 2.0})

    payload = model.to_payload()

    assert payload == {"strategy": "pareto", "centers": {"a": 1.5}, "scales": {"a": 2.0}}
    assert ScalingModel.from_payload(payload) == model


@pytest.mark.parametrize("key", ["strategy", "centers", "scales"])
def test_scaling_model_from_payload_rejects_missing_entry(key: str) -> None:
    """Raise ``KeyError`` when a payload entry is missing."""
    payload = ScalingModel(
        strategy="pareto", centers={"a": 1.5}, scales={"a": 2.0}
    ).to_payload()
    del payload[key]

    with pytest.raises(KeyError, match=key):
        ScalingModel.from_payload(payload)
