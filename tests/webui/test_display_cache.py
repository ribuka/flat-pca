"""Tests for the display cache of the exploration screens."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from flat_pca.webui.services.display_cache import DisplayCache, LruCache


def test_lru_cache_evicts_the_least_recently_used_entry() -> None:
    """A hit refreshes an entry; the oldest entry is evicted beyond the limit."""
    cache: LruCache[str, int] = LruCache(2)
    loads: list[str] = []

    def loader(key: str, value: int) -> Callable[[], int]:
        """Return a loader recording its calls."""

        def load() -> int:
            """Record the load and return the value."""
            loads.append(key)
            return value

        return load

    cache.get_or_load("a", loader("a", 1))
    cache.get_or_load("b", loader("b", 2))
    assert cache.get_or_load("a", loader("a", 0)) == 1
    cache.get_or_load("c", loader("c", 3))
    assert cache.get_or_load("b", loader("b", 4)) == 4

    assert loads == ["a", "b", "c", "b"]
    assert len(cache) == 2


def test_lru_cache_does_not_keep_failed_loads() -> None:
    """A loader's exception propagates and nothing is cached."""
    cache: LruCache[str, int] = LruCache(1)

    def fail() -> int:
        """Raise like a broken file."""
        raise ValueError("broken")

    with pytest.raises(ValueError, match="broken"):
        cache.get_or_load("a", fail)
    assert len(cache) == 0


def test_raw_spectra_adds_step_time_and_rereads_changed_files(tmp_path: Path) -> None:
    """Raw spectra get StepTime columns and are reloaded after a change."""
    path = tmp_path / "s.parquet"
    pl.DataFrame(
        {"Time": [1.0, 0.0], "Step": [1, 1], "Sequence": [1, 1], "400.0nm": [2.0, 1.0]}
    ).write_parquet(path)
    cache = DisplayCache()

    first = cache.raw_spectra(path)
    assert cache.raw_spectra(path) is first
    assert first["StepTime"].to_list() == [0.0, 1.0]

    pl.DataFrame(
        {"Time": [0.0], "Step": [1], "Sequence": [1], "400.0nm": [5.0]}
    ).write_parquet(path)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))

    assert cache.raw_spectra(path)["400.0nm"].to_list() == [5.0]
