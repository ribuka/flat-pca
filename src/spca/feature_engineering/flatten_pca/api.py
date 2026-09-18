"""Public API orchestration for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ..pca import PcaModel
from .downsampling import (
    apply_t_downsampling,
    apply_w_downsampling,
    collect_unique_times,
    collect_unique_wavelengths,
)
from .flatten import flatten_inputs
from .input import load_and_validate_inputs
from .normalization import apply_t_normalization, apply_w_normalization
from .pca_scores import fit_flattened_pca
from .smoothing import apply_t_smoothing, apply_w_smoothing


def flatten_pca(
    paths: Sequence[str | Path] | None = None,
    *,
    flattened: pl.LazyFrame | None = None,
    n_component: int | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
) -> PcaModel:
    """Preprocess, flatten, and fit PCA across spectral Parquet files.

    Parameters
    ----------
    paths : Sequence[str | Path] | None, default None
        One or more Parquet input paths. Exactly one of ``paths`` and
        ``flattened`` must be supplied.
    flattened : pl.LazyFrame | None, default None
        Existing flattened features used directly for PCA fitting.
    n_component : int | None, default None
        Number of PCA score columns to append. If ``None``, use the maximum
        available count: the smaller of the input-file count and flattened
        spectral-feature count.
    t_smoothing_window : float | None, default None
        Positive time-direction smoothing half-window, or ``None``.
    w_smoothing_window : float | None, default None
        Positive wavelength-direction smoothing half-window, or ``None``.
    t_normalization_range : tuple[float, float] | None, default None
        Inclusive time interval used for normalization, or ``None``.
    w_normalization_range : tuple[float, float] | None, default None
        Inclusive wavelength interval used for normalization, or ``None``.
    t_downsampling_stride : int, default 1
        Interval between retained values in the shared sorted Time array.
    w_downsampling_stride : int, default 1
        Interval between retained values in the shared sorted wavelength array.

    Returns
    -------
    PcaModel
        Fitted PCA pipeline state.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If inputs, preprocessing arguments, or ``n_component`` are invalid.
    """
    if (paths is None) == (flattened is None):
        raise ValueError("exactly one of paths or flattened must be specified")
    if flattened is None:
        assert paths is not None
        flattened = preprocess_and_flatten(
            paths,
            t_smoothing_window=t_smoothing_window,
            w_smoothing_window=w_smoothing_window,
            t_normalization_range=t_normalization_range,
            w_normalization_range=w_normalization_range,
            t_downsampling_stride=t_downsampling_stride,
            w_downsampling_stride=w_downsampling_stride,
        )
    return fit_flattened_pca(flattened, n_component)


def preprocess_and_flatten(
    paths: Sequence[str | Path],
    *,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
) -> pl.LazyFrame:
    """Build a lazy preprocessing and deterministic flattening query.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more Parquet input paths.
    t_smoothing_window, w_smoothing_window : float | None, default None
        Positive smoothing half-windows, or ``None`` to disable each stage.
    t_normalization_range, w_normalization_range : tuple[float, float] | None, default None
        Inclusive normalization ranges, or ``None`` to disable each stage.
    t_downsampling_stride, w_downsampling_stride : int, default 1
        Positive intervals in the shared sorted Time and wavelength arrays.

    Returns
    -------
    pl.LazyFrame
        Deferred one-row-per-file flattened features with ``filename`` first.
    """
    loaded_inputs = load_and_validate_inputs(paths)
    frames = [frame for _, frame in loaded_inputs]
    unique_times = collect_unique_times(frames)
    unique_wavelengths = collect_unique_wavelengths(frames)
    prepared_inputs: list[tuple[Path, pl.LazyFrame]] = []
    for path, frame in loaded_inputs:
        prepared = apply_t_smoothing(frame, t_smoothing_window)
        prepared = apply_w_smoothing(prepared, w_smoothing_window)
        prepared = apply_t_normalization(prepared, t_normalization_range)
        prepared = apply_w_normalization(prepared, w_normalization_range)
        prepared = apply_t_downsampling(
            prepared,
            unique_times,
            t_downsampling_stride,
        )
        prepared = apply_w_downsampling(
            prepared,
            unique_wavelengths,
            w_downsampling_stride,
        )
        prepared_inputs.append((path, prepared))

    flattened = flatten_inputs(prepared_inputs)
    assert isinstance(flattened, pl.LazyFrame)
    return flattened
