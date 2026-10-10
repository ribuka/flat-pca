"""Counting the values of a data table's columns for its column menus."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .config import TableConfig


@dataclass(frozen=True)
class ColumnCounts:
    """Numbers of rows shown beside the filters of a table's column menus.

    Attributes
    ----------
    values : dict[str, dict[str, int]]
        Rows of each non-null value (as text) of each ``"choice"`` column,
        keyed by column name.
    nulls : dict[str, int]
        Null values of each column with a filter, keyed by column name.
    """

    values: dict[str, dict[str, int]]
    nulls: dict[str, int]


def count_values(frame: pl.DataFrame | pl.LazyFrame, config: TableConfig) -> ColumnCounts:
    """Count the values of the ``"choice"`` columns and the nulls of the filtered ones.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table; the counts do not depend on the filters.
    config : TableConfig
        Table settings.

    Returns
    -------
    ColumnCounts
        The counts, collected in one query.
    """
    filtered = list(config.filtered_columns)
    choices = [
        name for name, column in config.filtered_columns.items() if column.filter == "choice"
    ]
    if not filtered:
        return ColumnCounts(values={}, nulls={})
    # Positional names keep the counts apart from any column name.
    counted = (
        frame.lazy()
        .select(
            *(
                pl.col(name)
                .cast(pl.String)
                .alias("value")
                .drop_nulls()
                .value_counts(name="n")
                .implode()
                .alias(f"values{index}")
                for index, name in enumerate(choices)
            ),
            *(
                pl.col(name).null_count().alias(f"nulls{index}")
                for index, name in enumerate(filtered)
            ),
        )
        .collect()
        .row(0, named=True)
    )
    return ColumnCounts(
        values={
            name: {pair["value"]: pair["n"] for pair in counted[f"values{index}"]}
            for index, name in enumerate(choices)
        },
        nulls={name: counted[f"nulls{index}"] for index, name in enumerate(filtered)},
    )
