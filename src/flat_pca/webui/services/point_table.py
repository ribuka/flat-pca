"""Table of the files behind the points of a figure, with their metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from ..templating import format_value
from .scored_samples import metadata_columns

STEM_HEADER = "file"
# Mode bar buttons of the figures whose points fill the table by a range.
SELECT_TOOLS = ("select2d", "lasso2d")


@dataclass(frozen=True)
class PointTable:
    """Formatted rows of the files drawn as points, one row per file.

    Attributes
    ----------
    columns : list[str]
        Header of each column: the file name, the metadata columns, and the
        extra columns.
    rows : list[list[str]]
        Cell texts of each file, in the order of the frame given; the first
        cell is the stem.
    """

    columns: list[str]
    rows: list[list[str]]


def point_table(
    samples: pl.DataFrame, extra: Mapping[str, Sequence[object]]
) -> PointTable:
    """Return the table rows of the files drawn as points.

    The page embeds every row, and the browser shows the rows of the chosen
    points.

    Parameters
    ----------
    samples : pl.DataFrame
        ``stem`` and the metadata columns of each file.
    extra : Mapping[str, Sequence[object]]
        Further columns keyed by header, one value per row of ``samples``,
        such as the statistics drawn in the figure.

    Returns
    -------
    PointTable
        Header and cell texts formatted by ``format_value``.

    Raises
    ------
    ValueError
        If an extra column does not have one value per file.
    """
    metadata = metadata_columns(samples)
    lengths = {header: len(values) for header, values in extra.items()}
    if any(length != samples.height for length in lengths.values()):
        raise ValueError(f"extra columns must have {samples.height} values: {lengths}")
    columns = [samples[column].to_list() for column in ("stem", *metadata)]
    columns += [list(values) for values in extra.values()]
    return PointTable(
        columns=[STEM_HEADER, *metadata, *extra],
        rows=[[format_value(value) for value in row] for row in zip(*columns, strict=True)],
    )
