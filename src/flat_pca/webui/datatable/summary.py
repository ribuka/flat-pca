"""What a data table shows over every row, whatever its state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .config import TableConfig
from .counts import ColumnCounts, count_values
from .formatting import dtype_label
from .histograms import ColumnHistogram, count_histograms


@dataclass(frozen=True)
class TableSummary:
    """The parts of a table view that depend only on the frame.

    The column menus and headers show them for every state alike, so an
    application whose frame changes seldom can make one per frame and pass
    it to ``apply_state`` for each request.

    Attributes
    ----------
    schema : dict[str, pl.DataType]
        Column types of the frame.
    options : dict[str, list[str]]
        Values offered by each ``"choice"`` filter.
    counts : ColumnCounts
        Rows of each value of the ``"choice"`` columns and null values of the
        filtered columns, over every row.
    types : dict[str, str]
        Type of each shown column (``dtype_label``), keyed by column name.
    histograms : dict[str, ColumnHistogram]
        Distribution of each shown column over every row
        (``count_histograms``, every bin in the filter); empty unless
        ``TableConfig.histograms``.
    """

    schema: dict[str, pl.DataType]
    options: dict[str, list[str]]
    counts: ColumnCounts
    types: dict[str, str]
    histograms: dict[str, ColumnHistogram]


def choice_options(
    frame: pl.DataFrame | pl.LazyFrame, config: TableConfig
) -> dict[str, list[str]]:
    """Return the distinct values of each ``"choice"`` column.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table.
    config : TableConfig
        Table settings.

    Returns
    -------
    dict[str, list[str]]
        Non-null values as sorted text, keyed by column name.
    """
    names = [
        column.name for column in config.filtered_columns.values() if column.filter == "choice"
    ]
    if not names:
        return {}
    values = frame.lazy().select(
        pl.col(name).cast(pl.String).drop_nulls().unique().sort().implode() for name in names
    ).collect()
    return {name: values[name][0].to_list() for name in names}


def summarize_table(
    frame: pl.DataFrame | pl.LazyFrame,
    config: TableConfig,
    options: Mapping[str, Sequence[str]] | None = None,
) -> TableSummary:
    """Count what a table shows over every row of its frame.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table in its default order.
    config : TableConfig
        Table settings.
    options : Mapping[str, Sequence[str]] | None, default None
        Values offered by each ``"choice"`` filter; ``choice_options`` of the
        frame if ``None``. A given value missing from the frame counts 0.

    Returns
    -------
    TableSummary
        The column types, choices, counts, and (with
        ``TableConfig.histograms``) histograms of the frame.
    """
    schema = dict(frame.lazy().collect_schema())
    return TableSummary(
        schema=schema,
        options=(
            choice_options(frame, config)
            if options is None
            else {name: list(values) for name, values in options.items()}
        ),
        counts=count_values(frame, config),
        types={column.name: dtype_label(schema[column.name]) for column in config.columns},
        histograms=count_histograms(frame, config, schema) if config.histograms else {},
    )
