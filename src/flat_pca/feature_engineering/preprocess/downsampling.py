"""Downsampling stages for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import polars as pl

from ..flatten_pca.schema import parse_wavelength, wavelength_columns


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
            for column in wavelength_columns(_column_names(frame))
        }
    )


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
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("t_downsampling_stride must be an integer of at least 1")

    selected_times = unique_times[:: int(stride)]
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
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("w_downsampling_stride must be an integer of at least 1")

    columns = _column_names(frame)
    spectra = wavelength_columns(columns)
    non_spectral_columns = [column for column in columns if column not in spectra]
    wavelength_to_column = {
        parse_wavelength(column): column for column in spectra
    }
    selected_columns = [
        wavelength_to_column[wavelength]
        for wavelength in unique_wavelengths[:: int(stride)]
    ]
    return frame.select(*non_spectral_columns, *selected_columns)


def _column_names(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return column names without materializing a LazyFrame.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Spectral frame whose schema supplies the column names.

    Returns
    -------
    list[str]
        Frame column names in their existing order.
    """
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns
