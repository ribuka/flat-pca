"""Step-row filtering for Flatten-PCA input frames."""

from __future__ import annotations

import polars as pl


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
