"""Tests for the display cache of the exploration screens."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flat_pca.webui.jobs.fit_run import COMPONENTS_FILE, build_fit_config, run_fit
from flat_pca.webui.services.display_cache import DisplayCache, LruCache, ShownMatrices
from flat_pca.webui.services.fit_artifacts import (
    RunArtifactError,
    load_display_artifacts,
)
from flat_pca.webui.services.spectral_matrix import SpectralMatrix
from flat_pca.webui.settings import Settings


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


def test_lru_cache_evicts_beyond_the_size_limit() -> None:
    """Oldest entries are evicted beyond ``max_bytes``; oversized values are not kept."""
    cache: LruCache[str, bytes] = LruCache(10, max_bytes=5, size_of=len)

    cache.put("a", b"aa")
    cache.put("b", b"bb")
    assert cache.get("a") == b"aa"
    cache.put("c", b"cc")

    assert cache.get("b") is None
    assert cache.get("a") == b"aa"
    assert cache.total_bytes == 4

    cache.put("a", b"toolarge")
    assert cache.get("a") is None
    assert cache.get("c") == b"cc"
    assert cache.total_bytes == 2


def test_lru_cache_requires_size_of_with_max_bytes() -> None:
    """A size limit without a way to measure values is rejected."""
    with pytest.raises(ValueError, match="together"):
        LruCache(1, max_bytes=1)


def test_shown_matrices_are_kept_by_their_choices() -> None:
    """Kept matrices are returned for the same key and replaced by a new one."""
    cache = DisplayCache()
    matrix = SpectralMatrix(
        values=np.zeros((2, 3)), wavelengths=np.arange(3.0), step_times=np.arange(2.0)
    )
    first = ShownMatrices(value_name="intensity", matrices={"a": matrix})
    second = ShownMatrices(value_name="residual", matrices={"a": matrix})

    assert cache.shown_matrices(("raw", "a")) is None
    cache.keep_shown_matrices(("raw", "a"), first)
    assert cache.shown_matrices(("raw", "a")) is first
    cache.keep_shown_matrices(("raw", "a"), second)
    assert cache.shown_matrices(("raw", "a")) is second
    assert first.nbytes == 6 * 8 + 3 * 8 + 2 * 8


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


def test_fit_artifacts_keep_matrices_memory_mapped(
    settings: Settings, spectra_paths: list[Path]
) -> None:
    """Display artifacts restore no model, so the components stay memory-mapped."""
    run_dir = _fit_run_dir(settings, spectra_paths)

    artifacts = DisplayCache().fit_artifacts(run_dir)

    assert isinstance(artifacts.x, np.memmap)
    assert isinstance(artifacts.components, np.memmap)
    assert artifacts.components.shape == (2, artifacts.features.height)


def test_fit_artifacts_reject_inconsistent_components(
    settings: Settings, spectra_paths: list[Path]
) -> None:
    """Components of another feature count are reported as unreadable artifacts."""
    run_dir = _fit_run_dir(settings, spectra_paths)
    np.save(run_dir / COMPONENTS_FILE, np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(RunArtifactError, match="components.npy"):
        load_display_artifacts(run_dir)


def _fit_run_dir(settings: Settings, paths: list[Path]) -> Path:
    """Run a two-component fit of the synthetic spectra and return its directory."""
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in paths],
        {"target_steps": [1, 2], "max_null_ratio": 0.1},
        {
            "n_component": 2,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "none",
        },
        {"cumulative_explained_variance": 1, "alpha": 0.01},
        {"cumulative_explained_variance": 1, "alpha": 0.01},
    )
    run_dir = settings.runs_dir / "fit"
    run_dir.mkdir(parents=True)
    run_fit(json.loads(json.dumps(config)), run_dir)
    return run_dir
