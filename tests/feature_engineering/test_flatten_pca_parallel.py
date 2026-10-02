"""Tests for per-file parallel execution of Flatten-PCA preprocessing (#30)."""

import threading
from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering import flatten_pca, preprocess_and_flatten
from flat_pca.feature_engineering.flatten_pca import parallel
from flat_pca.feature_engineering.flatten_pca.parallel import (
    run_per_file,
    validate_workers,
)

INVALID_WORKERS = [0, -1, 1.0, 2.5, "2", True, None]


@pytest.mark.parametrize("workers", [1, 2, 12])
def test_validate_workers_accepts_positive_integers(workers: int) -> None:
    """Return a positive integer worker count unchanged."""
    assert validate_workers(workers) == workers


@pytest.mark.parametrize("workers", INVALID_WORKERS)
def test_validate_workers_rejects_non_positive_or_non_integer(workers: object) -> None:
    """Reject zero, negatives, non-integers, bool, and None."""
    with pytest.raises(ValueError, match="workers must be an integer >= 1"):
        validate_workers(workers)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_run_per_file_returns_results_in_item_order(workers: int) -> None:
    """Return results in input order even when later items finish first."""
    first_may_finish = threading.Event()

    def work(item: int) -> int:
        if item == 0 and workers > 1:
            first_may_finish.wait(timeout=5)
        if item == 3:
            first_may_finish.set()
        return item * 10

    assert run_per_file(work, [0, 1, 2, 3], workers) == [0, 10, 20, 30]


def test_run_per_file_raises_earliest_failing_item_error() -> None:
    """Raise the earliest item's error even if a later item fails first."""
    later_failed = threading.Event()

    def work(item: int) -> int:
        if item == 0:
            later_failed.wait(timeout=5)
            raise ValueError("item 0")
        if item == 1:
            later_failed.set()
            raise ValueError("item 1")
        return item

    with pytest.raises(ValueError, match="item 0"):
        run_per_file(work, [0, 1, 2], workers=2)


def test_run_per_file_does_not_create_executor_for_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the plain sequential loop without a thread pool when workers is 1."""

    def fail_executor(*args: object, **kwargs: object) -> None:
        raise AssertionError("ThreadPoolExecutor must not be created")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", fail_executor)
    calling_thread = threading.get_ident()

    assert run_per_file(lambda item: (item, threading.get_ident()), [1, 2], 1) == [
        (1, calling_thread),
        (2, calling_thread),
    ]


def test_preprocess_and_flatten_matches_across_worker_counts(
    real_fixture_paths: list[Path],
) -> None:
    """Produce identical output for 1, 2, and 4 workers on real fixtures."""
    kwargs = {
        "t_smoothing_window": 10000.0,
        "w_smoothing_window": 50.0,
        "t_normalization_range": (0.0, 20000.0),
        "t_downsampling_stride": 2,
        "w_downsampling_stride": 2,
        "max_null_ratio": 1.0,
    }
    results = [
        preprocess_and_flatten(real_fixture_paths, workers=workers, **kwargs).collect()
        for workers in (1, 2, 4)
    ]

    assert results[0].height == len(real_fixture_paths)
    assert all(result.equals(results[0]) for result in results[1:])


def _write_invalid_copy(source: Path, destination: Path) -> Path:
    """Write a copy of ``source`` with one NaN spectral value."""
    frame = pl.read_parquet(source)
    wavelength = next(
        column for column in frame.columns if column not in {"Time", "Step", "Sequence"}
    )
    frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col(wavelength))
        .alias(wavelength)
    ).write_parquet(destination)
    return destination


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_preprocess_and_flatten_reports_first_invalid_file_regardless_of_workers(
    workers: int, tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Report the first invalid file in path order for every worker count."""
    first_invalid = _write_invalid_copy(real_fixture_paths[0], tmp_path / "a.parquet")
    second_invalid = _write_invalid_copy(real_fixture_paths[1], tmp_path / "b.parquet")
    paths = [*real_fixture_paths[2:], first_invalid, second_invalid]

    with pytest.raises(ValueError, match="null, NaN, or infinite") as error:
        preprocess_and_flatten(paths, workers=workers)

    assert first_invalid.name in str(error.value)
    assert second_invalid.name not in str(error.value)


@pytest.mark.parametrize("workers", INVALID_WORKERS)
def test_public_api_rejects_invalid_workers_before_reading_input(
    workers: object, tmp_path: Path
) -> None:
    """Raise ValueError for invalid workers before touching a missing path."""
    missing = [tmp_path / "missing.parquet"]

    with pytest.raises(ValueError, match="workers must be an integer >= 1"):
        preprocess_and_flatten(missing, workers=workers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workers must be an integer >= 1"):
        flatten_pca(missing, workers=workers)  # type: ignore[arg-type]


def test_legacy_path_ignores_workers(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Keep legacy-path results unchanged when workers is specified."""
    deferred = [
        preprocess_and_flatten(
            real_fixture_paths, materialize_once=False, workers=workers
        ).collect()
        for workers in (1, 4)
    ]
    assert deferred[1].equals(deferred[0])

    integer_path = tmp_path / "integer.parquet"
    pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Step": [0, 0],
            "Sequence": [0, 0],
            "500.0nm": pl.Series([1, 2], dtype=pl.Int64),
        }
    ).write_parquet(integer_path)
    fallback = [
        preprocess_and_flatten([integer_path], workers=workers).collect()
        for workers in (1, 4)
    ]
    assert fallback[1].equals(fallback[0])
