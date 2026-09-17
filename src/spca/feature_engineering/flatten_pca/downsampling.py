"""Downsampling stages for the Flatten-PCA workflow."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import polars as pl


def collect_unique_times(frames: Sequence[pl.DataFrame]) -> list[float]:
    """Collect sorted unique Time values across validated input frames.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame]
        Validated input frames containing numeric ``Time`` columns.

    Returns
    -------
    list[float]
        Unique Time values in numeric ascending order.
    """
    return (
        pl.concat(
            frame.select(pl.col("Time").cast(pl.Float64)) for frame in frames
        )
        .get_column("Time")
        .unique()
        .sort()
        .to_list()
    )


def apply_t_downsampling(
    frame: pl.DataFrame,
    unique_times: list[float],
    stride: int,
) -> pl.DataFrame:
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
