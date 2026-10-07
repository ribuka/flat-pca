"""Estimating the feature count and memory of a fit job from the catalog."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..database import Database

BYTES_PER_VALUE = 8
# Peak memory of a fit job as a multiple of N x F x 8 bytes, measured with
# run_fit on synthetic data (N = 300 and 600 files, F = 30,000 features) and
# rounded up from the larger case. The factors shrink as N grows because
# part of the memory does not scale with N.
NUMPY_PATH_MEMORY_FACTOR = 10.0
POLARS_PATH_MEMORY_FACTOR = 16.0
KMEANS_MEMORY_FACTOR = 24.0
GIB = 1024**3


@dataclass(frozen=True)
class FitEstimate:
    """Estimated size of a fit job.

    Attributes
    ----------
    n_files : int
        Number of target files ``N``.
    n_features : int
        Estimated flattened feature count ``F`` before sparse-column pruning.
    memory_factor : float
        Multiple of ``N * F * 8`` bytes used for the estimate.
    memory_bytes : int
        Estimated peak memory, ``memory_factor * N * F * 8`` bytes.
    """

    n_files: int
    n_features: int
    memory_factor: float
    memory_bytes: int

    @property
    def memory_gb(self) -> float:
        """Return the estimated memory in GiB.

        Returns
        -------
        float
            ``memory_bytes / 1024**3``.
        """
        return self.memory_bytes / GIB

    def exceeds(self, limit_gb: float) -> bool:
        """Return whether the estimate exceeds a limit.

        Parameters
        ----------
        limit_gb : float
            Limit in GiB, such as ``jobs.memory_warn_gb``.

        Returns
        -------
        bool
            ``True`` if ``memory_gb`` is above ``limit_gb``.
        """
        return self.memory_gb > limit_gb


def memory_factor(impute_strategy: object, scaling_strategy: object) -> float:
    """Return the peak-memory factor of a fit job's PCA pipeline.

    ``"drop"`` imputation with ``"none"`` scaling fits on the NumPy fast
    path; other combinations prepare the data with polars and need more
    memory, and ``"kmeans"`` imputation the most.

    Parameters
    ----------
    impute_strategy : object
        Imputation strategy; an unknown value gets the largest factor.
    scaling_strategy : object
        Scaling strategy.

    Returns
    -------
    float
        Multiple of ``N * F * 8`` bytes.
    """
    if impute_strategy == "drop" and scaling_strategy == "none":
        return NUMPY_PATH_MEMORY_FACTOR
    if impute_strategy in ("drop", "median"):
        return POLARS_PATH_MEMORY_FACTOR
    return KMEANS_MEMORY_FACTOR


def retained_row_count(
    n_rows: int, step_time_max: float, edge_trim: Sequence[float] | None
) -> int:
    """Estimate the rows of one ``(Step, Sequence)`` kept by edge trimming.

    Rows are assumed evenly spaced in ``StepTime`` from 0 to
    ``step_time_max``.

    Parameters
    ----------
    n_rows : int
        Rows of the segment.
    step_time_max : float
        Largest ``StepTime`` of the segment.
    edge_trim : Sequence[float] | None
        ``StepTime`` and ``ReverseStepTime`` thresholds, or ``None``.

    Returns
    -------
    int
        Estimated retained rows.
    """
    if edge_trim is None:
        return n_rows
    start, end = (max(float(bound), 0.0) for bound in edge_trim)
    if step_time_max <= 0 or n_rows <= 1:
        return n_rows if start <= 0 and end <= 0 else 0
    spacing = step_time_max / (n_rows - 1)
    kept_span = step_time_max - start - end
    if kept_span < 0:
        return 0
    first = math.ceil(start / spacing - 1e-9)
    last = math.floor((step_time_max - end) / spacing + 1e-9)
    return max(last - first + 1, 0)


def estimate_time_points(
    segments: Sequence[Mapping[str, object]],
    target_steps: Sequence[int] | None,
    edge_trim: Sequence[float] | None,
    t_downsampling_stride: int,
) -> int:
    """Estimate the ``(Step, Sequence, StepTime)`` points of the flattened grid.

    Each ``(Step, Sequence)`` contributes the largest row count among the
    files, which matches the union of the files' grids when they share one
    sampling interval.

    Parameters
    ----------
    segments : Sequence[Mapping[str, object]]
        Catalog ``segments`` rows of the target files, with ``step``,
        ``sequence``, ``n_rows``, and ``step_time_max``.
    target_steps : Sequence[int] | None
        Retained steps, or ``None`` for every step.
    edge_trim : Sequence[float] | None
        Edge-trim thresholds, or ``None``.
    t_downsampling_stride : int
        Time-direction downsampling stride.

    Returns
    -------
    int
        Estimated time points.
    """
    largest: dict[tuple[int, int], tuple[int, float]] = {}
    for row in segments:
        step = int(row["step"])  # type: ignore[arg-type]
        if target_steps is not None and step not in target_steps:
            continue
        key = (step, int(row["sequence"]))  # type: ignore[arg-type]
        n_rows = int(row["n_rows"])  # type: ignore[arg-type]
        step_time_max = float(row["step_time_max"])  # type: ignore[arg-type]
        previous = largest.get(key, (0, 0.0))
        largest[key] = (max(previous[0], n_rows), max(previous[1], step_time_max))
    return sum(
        math.ceil(retained_row_count(n_rows, step_time_max, edge_trim) / t_downsampling_stride)
        for n_rows, step_time_max in largest.values()
    )


def estimate_wavelength_count(
    files: Sequence[Mapping[str, object]],
    wavelength_range: Sequence[float] | None,
    w_downsampling_stride: int,
) -> int:
    """Estimate the wavelength columns kept by range filtering and downsampling.

    The file with the most wavelengths is assumed to hold evenly spaced
    wavelengths between its minimum and maximum.

    Parameters
    ----------
    files : Sequence[Mapping[str, object]]
        Catalog ``files`` rows of the target files, with ``n_wavelengths``,
        ``wavelength_min``, and ``wavelength_max``.
    wavelength_range : Sequence[float] | None
        Inclusive retained wavelength range, or ``None``.
    w_downsampling_stride : int
        Wavelength-direction downsampling stride.

    Returns
    -------
    int
        Estimated wavelength columns.
    """
    if not files:
        return 0
    widest = max(files, key=lambda row: int(row["n_wavelengths"]))  # type: ignore[arg-type]
    count = int(widest["n_wavelengths"])  # type: ignore[arg-type]
    lowest = float(widest["wavelength_min"])  # type: ignore[arg-type]
    highest = float(widest["wavelength_max"])  # type: ignore[arg-type]
    if wavelength_range is not None and count > 1 and highest > lowest:
        lower, upper = (float(bound) for bound in wavelength_range)
        spacing = (highest - lowest) / (count - 1)
        first = max(math.ceil((lower - lowest) / spacing - 1e-9), 0)
        last = min(math.floor((upper - lowest) / spacing + 1e-9), count - 1)
        count = max(last - first + 1, 0)
    elif wavelength_range is not None:
        lower, upper = (float(bound) for bound in wavelength_range)
        count = count if lower <= lowest and highest <= upper else 0
    return math.ceil(count / w_downsampling_stride)


def estimate_fit(
    database: Database,
    stems: Sequence[str],
    preprocess: Mapping[str, object],
    pca: Mapping[str, object],
) -> FitEstimate:
    """Estimate the feature count and peak memory of a fit job.

    Parameters
    ----------
    database : Database
        Workspace database holding the catalog.
    stems : Sequence[str]
        Target file stems.
    preprocess : Mapping[str, object]
        ``preprocess_and_flatten`` keyword arguments; ``target_steps``,
        ``edge_trim``, ``wavelength_range``, and the downsampling strides
        are used, and missing or ``None`` entries leave their stage out.
    pca : Mapping[str, object]
        PCA settings; ``impute_strategy`` and ``scaling_strategy`` choose
        the memory factor (see ``memory_factor``).

    Returns
    -------
    FitEstimate
        Estimated feature count and memory.
    """
    stem_list = list(stems)
    files = database.fetch_dicts(
        "SELECT n_wavelengths, wavelength_min, wavelength_max FROM files "
        "WHERE list_contains(?, stem)",
        [stem_list],
    )
    segments = database.fetch_dicts(
        "SELECT step, sequence, n_rows, step_time_max FROM segments "
        "WHERE list_contains(?, stem)",
        [stem_list],
    )
    n_features = estimate_time_points(
        segments,
        preprocess.get("target_steps"),  # type: ignore[arg-type]
        preprocess.get("edge_trim"),  # type: ignore[arg-type]
        int(preprocess.get("t_downsampling_stride") or 1),  # type: ignore[arg-type]
    ) * estimate_wavelength_count(
        files,
        preprocess.get("wavelength_range"),  # type: ignore[arg-type]
        int(preprocess.get("w_downsampling_stride") or 1),  # type: ignore[arg-type]
    )
    n_files = len(files)
    factor = memory_factor(pca.get("impute_strategy"), pca.get("scaling_strategy"))
    return FitEstimate(
        n_files=n_files,
        n_features=n_features,
        memory_factor=factor,
        memory_bytes=math.ceil(factor * n_files * n_features * BYTES_PER_VALUE),
    )
