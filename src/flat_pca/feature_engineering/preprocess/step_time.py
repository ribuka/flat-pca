"""StepTime and ReverseStepTime column generation."""

from __future__ import annotations

import polars as pl

from flat_pca.utils import get_columns_from_polars


def add_step_time_columns(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Insert ``StepTime`` and ``ReverseStepTime`` columns after ``Time``.

    Rows are grouped into contiguous segments by sorting on ``Time`` and
    starting a new segment whenever ``Step`` or ``Sequence`` changes from the
    immediately preceding row. Within each segment, ``StepTime`` is the
    elapsed time since the segment's first row (``0.00`` for that row), and
    ``ReverseStepTime`` is the same segment values in reverse order, so the
    segment's last row always has ``ReverseStepTime`` equal to ``0.00``.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Step-filtered Flatten-PCA input frame containing ``Time``, ``Step``,
        and ``Sequence`` columns.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Frame with ``StepTime`` and ``ReverseStepTime`` inserted immediately
        after ``Time``, sorted by ``Time`` ascending.
    """
    columns = get_columns_from_polars(frame)
    time_index = columns.index("Time")
    ordered_columns = (
        columns[: time_index + 1]
        + ["StepTime", "ReverseStepTime"]
        + columns[time_index + 1 :]
    )

    sorted_frame = frame.sort("Time")
    is_new_segment = (
        (pl.col("Step") != pl.col("Step").shift(1))
        | (pl.col("Sequence") != pl.col("Sequence").shift(1))
    ).fill_null(True)
    segment_id = is_new_segment.cast(pl.Int64).cum_sum()

    with_segment = sorted_frame.with_columns(segment_id.alias("_step_time_segment"))
    with_step_time = with_segment.with_columns(
        (pl.col("Time") - pl.col("Time").min().over("_step_time_segment")).alias(
            "StepTime"
        ),
        (pl.col("Time").max().over("_step_time_segment") - pl.col("Time")).alias(
            "ReverseStepTime"
        ),
    ).drop("_step_time_segment")

    return with_step_time.select(ordered_columns)
