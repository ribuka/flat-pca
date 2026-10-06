"""Round-trip tests for saved transform payloads."""

import json
from pathlib import Path

import polars as pl
import polars.testing
import pytest

from flat_pca.feature_engineering.pca import PcaModel, transform_pca

PAYLOAD_FIXTURE_DIRECTORY = Path("tests/fixtures/pca_transform_payload")
PAYLOAD_FIXTURES = [
    "median_winsorize_robust",
    "kmeans_drop_pareto",
    "drop_none_none",
]


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


@pytest.mark.parametrize("name", PAYLOAD_FIXTURES)
def test_saved_payload_reproduces_saved_transform(name: str) -> None:
    """Restore a saved payload and reproduce the transform saved with it."""
    fixture = _load_fixture(name)
    model = PcaModel.from_transform_json(json.dumps(fixture["payload"]))

    actual = transform_pca(pl.DataFrame(fixture["input"]).lazy(), model).collect()

    polars.testing.assert_frame_equal(actual, pl.DataFrame(fixture["expected"]))


@pytest.mark.parametrize("name", PAYLOAD_FIXTURES)
def test_saved_payload_round_trips_unchanged(name: str) -> None:
    """Write back a restored payload with the same keys, order, and values."""
    payload = _load_fixture(name)["payload"]
    model = PcaModel.from_transform_json(json.dumps(payload))

    written = json.loads(model.to_transform_json())

    assert list(written) == list(payload)
    assert written == payload
