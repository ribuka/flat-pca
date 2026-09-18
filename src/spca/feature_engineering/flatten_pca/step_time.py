"""Step-row filtering, StepTime/ReverseStepTime generation, and edge trimming."""

from __future__ import annotations

import polars as pl

from .schema import wavelength_columns


def filter_target_steps(
    frame: pl.DataFrame | pl.LazyFrame,
    target_steps: list[int] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Keep only rows whose ``Step`` value is in ``target_steps``.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Validated Flatten-PCA input frame.
    target_steps : list[int] | None
        ``Step`` values to keep. If ``None``, no row filtering is applied.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Frame containing only rows whose ``Step`` is in ``target_steps``, or
        the input frame unchanged if ``target_steps`` is ``None``.
    """
    if target_steps is None:
        return frame
    return frame.filter(pl.col("Step").is_in(target_steps))


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
    columns = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )
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


def apply_edge_trim(
    frame: pl.DataFrame | pl.LazyFrame,
    edge_trim: list[float, float] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Null out non-meta values for rows near a segment's start or end.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame containing ``StepTime`` and ``ReverseStepTime`` columns.
    edge_trim : list[float, float] | None
        ``(edge_trim[0], edge_trim[1])`` thresholds. Rows with ``StepTime``
        less than ``edge_trim[0]`` or ``ReverseStepTime`` less than
        ``edge_trim[1]`` have every non-meta column overwritten with
        ``None``. Meta columns (``Time``, ``Step``, ``StepTime``,
        ``ReverseStepTime``, ``Sequence``) are never overwritten. If
        ``None``, the frame is returned unchanged.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Frame with non-meta columns nulled on rows within either trimmed
        edge, or the input frame unchanged if ``edge_trim`` is ``None``.
    """
    if edge_trim is None:
        return frame
    start_threshold, end_threshold = edge_trim

    columns = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )
    spectra = wavelength_columns(columns)
    condition = (pl.col("StepTime") < start_threshold) | (
        pl.col("ReverseStepTime") < end_threshold
    )
    return frame.with_columns(
        pl.when(condition).then(None).otherwise(pl.col(column)).alias(column)
        for column in spectra
    )
