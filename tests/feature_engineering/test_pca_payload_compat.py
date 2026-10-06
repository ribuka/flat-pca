"""Compatibility tests for transform payloads saved in the current format."""

import json
from pathlib import Path

import polars as pl
import polars.testing
import pytest

from flat_pca.feature_engineering.pca import PcaModel, transform_pca

PAYLOAD_FIXTURE_DIRECTORY = Path("tests/fixtures/pca_transform_payload")
CURRENT_FORMAT_FIXTURES = [
    "median_winsorize_robust",
    "kmeans_drop_pareto",
    "drop_none_none",
]
LEGACY_FORMAT_FIXTURES = ["legacy_median_zscore"]


def _load_fixture(name: str) -> dict[str, dict[str, object]]:
    """Load one saved payload fixture.

    Parameters
    ----------
    name : str
        Fixture file name without the ``.json`` suffix.

    Returns
    -------
    dict[str, dict[str, object]]
        The ``input``, ``payload``, and ``expected`` entries of the fixture.
    """
    path = PAYLOAD_FIXTURE_DIRECTORY / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "name", [*CURRENT_FORMAT_FIXTURES, *LEGACY_FORMAT_FIXTURES]
)
def test_saved_payload_reproduces_saved_transform(name: str) -> None:
    """Restore a saved payload and reproduce the transform saved with it."""
    fixture = _load_fixture(name)
    model = PcaModel.from_transform_json(json.dumps(fixture["payload"]))

    actual = transform_pca(pl.DataFrame(fixture["input"]).lazy(), model).collect()

    polars.testing.assert_frame_equal(actual, pl.DataFrame(fixture["expected"]))


@pytest.mark.parametrize("name", CURRENT_FORMAT_FIXTURES)
def test_saved_payload_round_trips_unchanged(name: str) -> None:
    """Write back a restored payload with the same keys, order, and values."""
    payload = _load_fixture(name)["payload"]
    model = PcaModel.from_transform_json(json.dumps(payload))

    written = json.loads(model.to_transform_json())

    assert list(written) == list(payload)
    assert written == payload


def test_legacy_payload_restores_defaults() -> None:
    """Fill entries missing from a legacy payload with their defaults."""
    payload = _load_fixture("legacy_median_zscore")["payload"]

    model = PcaModel.from_transform_json(json.dumps(payload))

    assert model.impute_strategy == "median"
    assert model.impute_kmeans_n_clusters is None
    assert model.impute_kmeans_centroids == []
    assert model.outlier_strategy is None
    assert model.iqr_multiplier == 1.5
    assert model.outlier_lower_bounds == {}
    assert model.outlier_upper_bounds == {}
    assert model.winsor_lower_bounds == {}
    assert model.winsor_upper_bounds == {}
