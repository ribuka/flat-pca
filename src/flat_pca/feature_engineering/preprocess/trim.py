"""Edge trimming of non-meta spectral columns near segment boundaries."""

from __future__ import annotations

from typing import Literal

import polars as pl

from ..flatten_pca.schema import wavelength_columns


def apply_edge_trim(
    frame: pl.DataFrame | pl.LazyFrame,
    edge_trim: list[float, float] | None,
    action: Literal["drop", "null"] = "drop",
) -> pl.DataFrame | pl.LazyFrame:
    """Drop or null rows near a segment's start or end.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame containing ``StepTime`` and ``ReverseStepTime`` columns.
    edge_trim : list[float, float] | None
        ``(edge_trim[0], edge_trim[1])`` thresholds. Rows with ``StepTime``
        less than ``edge_trim[0]`` or ``ReverseStepTime`` less than
        ``edge_trim[1]`` are trimmed. If ``None``, the frame is returned
        unchanged.
    action : {"drop", "null"}, default "drop"
        ``"drop"`` removes trimmed rows. ``"null"`` overwrites every
        non-meta column in trimmed rows with ``None`` while preserving meta
        columns (``Time``, ``Step``, ``StepTime``, ``ReverseStepTime``,
        ``Sequence``).

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Frame with the requested trimming applied, or the input frame
        unchanged if ``edge_trim`` is ``None``.

    Raises
    ------
    ValueError
        If ``action`` is not ``"drop"`` or ``"null"``.
    """
    if action not in {"drop", "null"}:
        raise ValueError("action must be either 'drop' or 'null'")
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
    if action == "drop":
        return frame.filter(~condition)
    if action == "null":
        return frame.with_columns(
            pl.when(condition).then(None).otherwise(pl.col(column)).alias(column)
            for column in spectra
        )
    raise AssertionError("validated action must be 'drop' or 'null'")
