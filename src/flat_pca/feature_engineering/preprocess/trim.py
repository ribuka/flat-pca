"""Edge trimming of non-meta spectral columns near segment boundaries."""

from __future__ import annotations

import polars as pl

from ..flatten_pca.schema import wavelength_columns


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
