"""Tests for estimating the size of a fit job from the catalog."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from flat_pca.webui.services.fit_estimate import (
    BYTES_PER_VALUE,
    GIB,
    KMEANS_MEMORY_FACTOR,
    NUMPY_PATH_MEMORY_FACTOR,
    POLARS_PATH_MEMORY_FACTOR,
    FitEstimate,
    estimate_fit,
    estimate_time_points,
    estimate_wavelength_count,
    memory_factor,
    retained_row_count,
)
from flat_pca.webui.workspace import Workspace

SEGMENTS = [
    {"step": 1, "sequence": 1, "n_rows": 11, "step_time_max": 10.0},
    {"step": 1, "sequence": 1, "n_rows": 9, "step_time_max": 8.0},
    {"step": 2, "sequence": 1, "n_rows": 5, "step_time_max": 4.0},
    {"step": 2, "sequence": 2, "n_rows": 1, "step_time_max": 0.0},
]
FILES = [
    {"n_wavelengths": 101, "wavelength_min": 400.0, "wavelength_max": 500.0},
    {"n_wavelengths": 51, "wavelength_min": 400.0, "wavelength_max": 450.0},
]


@pytest.mark.parametrize(
    ("edge_trim", "expected"),
    [
        (None, 11),
        ((0.0, 0.0), 11),
        ((2.0, 0.0), 9),
        ((2.5, 3.0), 5),
        ((-1.0, -1.0), 11),
        ((6.0, 6.0), 0),
    ],
)
def test_retained_row_count_follows_the_step_time_grid(
    edge_trim: tuple[float, float] | None, expected: int
) -> None:
    """Rows evenly spaced over StepTime 0..10 are trimmed at both ends."""
    assert retained_row_count(11, 10.0, edge_trim) == expected


def test_single_row_segments_are_trimmed_by_any_positive_threshold() -> None:
    """A one-row segment has StepTime 0 at both ends."""
    assert retained_row_count(1, 0.0, None) == 1
    assert retained_row_count(1, 0.0, (0.0, 0.0)) == 1
    assert retained_row_count(1, 0.0, (0.5, 0.0)) == 0


def test_time_points_take_the_largest_segment_of_each_step_sequence() -> None:
    """Each (Step, Sequence) counts its largest file's rows."""
    assert estimate_time_points(SEGMENTS, None, None, 1) == 11 + 5 + 1
    assert estimate_time_points(SEGMENTS, [2], None, 1) == 5 + 1
    assert estimate_time_points(SEGMENTS, None, None, 2) == 6 + 3 + 1
    assert estimate_time_points(SEGMENTS, [1], (2.0, 0.0), 1) == 9


def test_wavelength_count_follows_the_range_and_stride() -> None:
    """The widest file's evenly spaced wavelengths are filtered and thinned."""
    assert estimate_wavelength_count(FILES, None, 1) == 101
    assert estimate_wavelength_count(FILES, (420.0, 439.5), 1) == 20
    assert estimate_wavelength_count(FILES, (300.0, 900.0), 1) == 101
    assert estimate_wavelength_count(FILES, None, 3) == 34
    assert estimate_wavelength_count(FILES, (600.0, 700.0), 1) == 0
    assert estimate_wavelength_count([], None, 1) == 0


@pytest.mark.parametrize(
    ("impute_strategy", "scaling_strategy", "expected"),
    [
        ("drop", "none", NUMPY_PATH_MEMORY_FACTOR),
        ("drop", "pareto", POLARS_PATH_MEMORY_FACTOR),
        ("median", "none", POLARS_PATH_MEMORY_FACTOR),
        ("kmeans", "none", KMEANS_MEMORY_FACTOR),
        ("unknown", "none", KMEANS_MEMORY_FACTOR),
    ],
)
def test_memory_factor_depends_on_the_fitting_path(
    impute_strategy: str, scaling_strategy: str, expected: float
) -> None:
    """The NumPy fast path needs the least memory and kmeans the most."""
    assert memory_factor(impute_strategy, scaling_strategy) == expected


def test_estimate_exceeds_compares_gib() -> None:
    """``exceeds`` compares the estimate in GiB with the limit."""
    estimate = FitEstimate(n_files=1, n_features=1, memory_factor=1.0, memory_bytes=2 * GIB)

    assert estimate.memory_gb == 2.0
    assert estimate.exceeds(1.5)
    assert not estimate.exceeds(2.0)


def test_estimate_fit_reads_the_catalog(
    workspace: Workspace, wait_for: Callable[..., dict[str, object]]
) -> None:
    """The fixture catalog gives 7 time points x 3 wavelengths per file."""
    assert wait_for(workspace.database, workspace.submit_catalog())["status"] == "succeeded"
    stems = ["run-1", "run-2", "run-10"]

    estimate = estimate_fit(
        workspace.database,
        stems,
        {"target_steps": [1, 2], "wavelength_range": None},
        {"impute_strategy": "median", "scaling_strategy": "none"},
    )

    assert (estimate.n_files, estimate.n_features) == (3, 21)
    assert estimate.memory_factor == POLARS_PATH_MEMORY_FACTOR
    assert estimate.memory_bytes == POLARS_PATH_MEMORY_FACTOR * 3 * 21 * BYTES_PER_VALUE
    only_step_1 = estimate_fit(
        workspace.database,
        stems[:1],
        {"target_steps": [1], "w_downsampling_stride": 2},
        {"impute_strategy": "drop", "scaling_strategy": "none"},
    )
    assert (only_step_1.n_files, only_step_1.n_features) == (1, 3 * 2)
