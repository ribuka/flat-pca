"""Edge trimming of non-meta spectral columns near segment boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl

from flat_pca.spectral.schema import wavelength_columns
from flat_pca.utils import get_columns_from_polars

from .ranges import validate_finite_bounds


def apply_edge_trim(
    frame: pl.DataFrame | pl.LazyFrame,
    edge_trim: Sequence[float] | None,
    action: Literal["drop", "null"] = "drop",
) -> pl.DataFrame | pl.LazyFrame:
    """Drop or null rows near a segment's start or end.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame containing ``StepTime`` and ``ReverseStepTime`` columns.
    edge_trim : Sequence[float] | None
        Two finite thresholds, as a ``tuple`` or ``list``. Rows with
        ``StepTime`` less than ``edge_trim[0]`` or ``ReverseStepTime`` less
        than ``edge_trim[1]`` are trimmed. The two thresholds are
        independent, so they need not be ordered. If ``None``, the frame is
        returned unchanged.
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
        If ``action`` is not ``"drop"`` or ``"null"``, or if ``edge_trim`` is
        not a pair of finite thresholds.
    """
    if action not in {"drop", "null"}:
        raise ValueError("action must be either 'drop' or 'null'")
    if edge_trim is None:
        return frame
    start_threshold, end_threshold = validate_finite_bounds(edge_trim, "edge_trim")

    columns = get_columns_from_polars(frame)
    spectra = wavelength_columns(columns)
    condition = (pl.col("StepTime") < start_threshold) | (
        pl.col("ReverseStepTime") < end_threshold
    )
    if action == "drop":
        return frame.filter(~condition)
    return frame.with_columns(
        pl.when(condition).then(None).otherwise(pl.col(column)).alias(column)
        for column in spectra
    )
