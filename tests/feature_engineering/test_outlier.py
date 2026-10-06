"""Tests for IQR-based outlier handling."""

import polars as pl

from flat_pca.feature_engineering.outlier import (
    OutlierBounds,
    OutlierModel,
    fit_outlier,
)

COLUMNS = ["a"]


def _training_frame() -> pl.DataFrame:
    """Return a column whose last value is far outside its IQR bounds.

    Returns
    -------
    pl.DataFrame
        Training data with one outlier row.
    """
    return pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]})


def test_drop_apply_reuses_fitted_bounds() -> None:
    """Judge new rows against the bounds fitted on the training data."""
    prepared, model = fit_outlier(_training_frame().lazy(), COLUMNS, "drop", 1.5)

    result = model.apply(pl.DataFrame({"a": [4.0, 50.0]}).lazy(), COLUMNS).collect()

    assert prepared.collect()["a"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert result["a"].to_list() == [4.0]


def test_winsorize_apply_clips_to_fitted_bounds() -> None:
    """Clip new outlier values to the fitted percentile bounds."""
    _, model = fit_outlier(_training_frame().lazy(), COLUMNS, "winsorize", 1.5)

    result = model.apply(pl.DataFrame({"a": [4.0, 500.0]}).lazy(), COLUMNS).collect()

    assert result["a"].to_list() == [4.0, model.bounds.winsor_upper["a"]]


def test_none_strategy_leaves_data_and_bounds_empty() -> None:
    """Fit no bounds and leave the data unchanged without a strategy."""
    frame = _training_frame()
    prepared, model = fit_outlier(frame.lazy(), COLUMNS, None, 2.0)

    assert model == OutlierModel(None, 2.0, OutlierBounds.empty())
    assert prepared.collect().equals(frame)
    assert model.apply(frame.lazy(), COLUMNS).collect().equals(frame)


def test_payload_round_trip() -> None:
    """Restore the same model from its flat payload entries."""
    _, model = fit_outlier(_training_frame().lazy(), COLUMNS, "winsorize", 2.0)

    payload = model.to_payload()

    assert list(payload) == [
        "outlier_strategy",
        "iqr_multiplier",
        "outlier_lower_bounds",
        "outlier_upper_bounds",
        "winsor_lower_bounds",
        "winsor_upper_bounds",
    ]
    assert OutlierModel.from_payload(payload) == model


def test_legacy_payload_restores_no_handling() -> None:
    """Read a ``"none"`` strategy and missing entries as no handling."""
    model = OutlierModel.from_payload({"outlier_strategy": "none"})

    assert model == OutlierModel(None, 1.5, OutlierBounds.empty())
