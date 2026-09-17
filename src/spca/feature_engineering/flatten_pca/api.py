"""Public API orchestration for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import polars as pl

from .downsampling import (
    apply_t_downsampling,
    apply_w_downsampling,
    collect_unique_times,
    collect_unique_wavelengths,
)
from .flatten import flatten_inputs
from .input import load_and_validate_inputs
from .normalization import apply_t_normalization, apply_w_normalization
from .smoothing import apply_t_smoothing, apply_w_smoothing


def flatten_pca(
    paths: Sequence[str | Path],
    *,
    n_component: int | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
) -> pl.DataFrame:
    """Preprocess, flatten, and fit PCA across spectral Parquet files.

    Parameters
    ----------
    paths : Sequence[str | Path]
        One or more Parquet input paths.
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
    pl.DataFrame
        One row per input file containing ``filename``, deterministic flattened
        features, and PCA score columns named ``pca-1`` through the requested
        or automatically selected component count.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.
    ValueError
        If inputs, preprocessing arguments, or ``n_component`` are invalid.
    """
    flattened = preprocess_and_flatten(
        paths,
        t_smoothing_window=t_smoothing_window,
        w_smoothing_window=w_smoothing_window,
        t_normalization_range=t_normalization_range,
        w_normalization_range=w_normalization_range,
        t_downsampling_stride=t_downsampling_stride,
        w_downsampling_stride=w_downsampling_stride,
    ).collect()
    feature_columns = flattened.columns[1:]
    max_component = min(flattened.height, len(feature_columns))
    if n_component is None:
        resolved_n_component = max_component
    elif (
        isinstance(n_component, bool)
        or not isinstance(n_component, Integral)
        or not 1 <= n_component <= max_component
    ):
        raise ValueError(
            f"n_component must be an integer between 1 and {max_component}"
        )
    else:
        resolved_n_component = int(n_component)

    from ..pca import fit_and_transform_pca

    return fit_and_transform_pca(
        df=flattened.lazy(),
        columns=feature_columns,
        n_component=resolved_n_component,
        max_n_component=None,
        impute_strategy="drop",
        outlier_strategy=None,
        scaling_strategy="none",
    ).collect()


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
