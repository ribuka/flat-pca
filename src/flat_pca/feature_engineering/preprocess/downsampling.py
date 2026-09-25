"""Downsampling stages for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import polars as pl

from flat_pca.spectral.schema import parse_wavelength, wavelength_columns
from flat_pca.utils import get_columns_from_polars


def collect_unique_times(
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
) -> list[float]:
    """Collect sorted unique Time values across validated input frames.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame | pl.LazyFrame]
        Validated input frames containing numeric ``Time`` columns.

    Returns
    -------
    list[float]
        Unique Time values in numeric ascending order.
    """
    return (
        pl.concat(
            frame.select(pl.col("Time").cast(pl.Float64)).collect()
            if isinstance(frame, pl.LazyFrame)
            else frame.select(pl.col("Time").cast(pl.Float64))
            for frame in frames
        )
        .get_column("Time")
        .unique()
        .sort()
        .to_list()
    )


def collect_unique_wavelengths(
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
) -> list[float]:
    """Collect sorted unique wavelengths across validated input frames.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame | pl.LazyFrame]
        Validated input frames with a shared wavelength-column set.

    Returns
    -------
    list[float]
        Unique wavelengths in numeric ascending order.
    """
    return sorted(
        {
            parse_wavelength(column)
            for frame in frames
            for column in wavelength_columns(get_columns_from_polars(frame))
        }
    )


def resolve_downsampled_values(
    unique_values: list[float],
    stride: int,
    argument_name: str,
) -> list[float]:
    """Validate a downsampling stride and select its regularly spaced values.

    Shared by the Time- and wavelength-direction downsampling stages, and by
    the NumPy fast path, so the stride validation rule and selection formula
    have one implementation.

    Parameters
    ----------
    unique_values : list[float]
        Numerically sorted unique values to downsample.
    stride : int
        Positive, non-boolean interval between retained indices.
    argument_name : str
        Public argument name used in the validation message.

    Returns
    -------
    list[float]
        Every ``stride``-th value of ``unique_values``, starting from index 0.

    Raises
    ------
    ValueError
        If ``stride`` is not a positive, non-boolean integer.
    """
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError(f"{argument_name} must be an integer of at least 1")
    return unique_values[:: int(stride)]


def apply_t_downsampling(
    frame: pl.DataFrame | pl.LazyFrame,
    unique_times: list[float],
    stride: int,
) -> pl.DataFrame | pl.LazyFrame:
    """Keep rows at regularly spaced indices in the shared Time values.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated spectral frame to downsample.
    unique_times : list[float]
        Numerically sorted unique Time values shared by the input collection.
    stride : int
        Positive, non-boolean interval between retained Time indices.

    Returns
    -------
    pl.DataFrame
        Frame containing every row whose Time is at a selected index.

    Raises
    ------
    ValueError
        If ``stride`` is not a positive, non-boolean integer.
    """
    selected_times = resolve_downsampled_values(
        unique_times, stride, "t_downsampling_stride"
    )
    return frame.filter(pl.col("Time").is_in(selected_times))


def apply_w_downsampling(
    frame: pl.DataFrame | pl.LazyFrame,
    unique_wavelengths: list[float],
    stride: int,
) -> pl.DataFrame | pl.LazyFrame:
    """Keep wavelength columns at regularly spaced shared indices.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated spectral frame to downsample.
    unique_wavelengths : list[float]
        Numerically sorted unique wavelengths shared by the input collection.
    stride : int
        Positive, non-boolean interval between retained wavelength indices.

    Returns
    -------
    pl.DataFrame
        Frame containing all metadata, StepTime columns (if present), and
        wavelength columns at selected indices.

    Raises
    ------
    ValueError
        If ``stride`` is not a positive, non-boolean integer.
    """
    selected_wavelengths = resolve_downsampled_values(
        unique_wavelengths, stride, "w_downsampling_stride"
    )

    columns = get_columns_from_polars(frame)
    spectra = wavelength_columns(columns)
    non_spectral_columns = [column for column in columns if column not in spectra]
    wavelength_to_column = {
        parse_wavelength(column): column for column in spectra
    }
    selected_columns = [
        wavelength_to_column[wavelength] for wavelength in selected_wavelengths
    ]
    return frame.select(*non_spectral_columns, *selected_columns)
