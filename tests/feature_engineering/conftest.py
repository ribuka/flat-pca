"""Shared fixtures for feature-engineering tests."""

from pathlib import Path

import pytest

FIXTURE_DIRECTORY = Path("tests/fixtures/real_subset")


@pytest.fixture(scope="session")
def real_fixture_paths() -> list[Path]:
    """Return real spectral Parquet fixture paths in filename order.

    Returns
    -------
    list[Path]
        Sorted paths under ``tests/fixtures/real_subset``.
    """
    return sorted(FIXTURE_DIRECTORY.glob("*.parquet"), key=lambda path: path.name)
