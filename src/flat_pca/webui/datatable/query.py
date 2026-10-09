"""Applying a table state to a polars frame: filtering, sorting, and paging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .config import TableConfig
from .pagination import Page, paginate
from .state import TableState


@dataclass(frozen=True)
class TableView:
    """What one page of a data table shows.

    Attributes
    ----------
    state : TableState
        State the view was made for.
    page : Page
        Position of the shown page; a page past the last one shows the last.
    rows : list[dict[str, object]]
        Rows of the shown page with every frame column, in display order.
    matching_keys : list[str]
        Keys of every row matching the filters, on all pages, as text; empty
        for a table that is not selectable.
    options : dict[str, list[str]]
        Values offered by each ``"choice"`` filter.
    """

    state: TableState
    page: Page
    rows: list[dict[str, object]]
    matching_keys: list[str]
    options: dict[str, list[str]]


def filter_expression(state: TableState) -> pl.Expr:
    """Return the expression selecting the rows that match a state's filters.

    Parameters
    ----------
    state : TableState
        State whose column names have been validated, for example by
        ``parse_state``.

    Returns
    -------
    pl.Expr
        Boolean expression combining every filter with AND. A text filter
        matches a case-insensitive substring; null values match no filter.
    """
    conditions = [pl.lit(True)]
    for name, text in state.text.items():
        conditions.append(
            pl.col(name)
            .cast(pl.String)
            .str.to_lowercase()
            .str.contains(text.lower(), literal=True)
        )
    for name, value in state.equals.items():
        conditions.append(pl.col(name).cast(pl.String) == value)
    for name, (lower, upper) in state.ranges.items():
        if lower is not None:
            conditions.append(pl.col(name) >= pl.lit(lower))
        if upper is not None:
            conditions.append(pl.col(name) <= pl.lit(upper))
    return pl.all_horizontal(conditions).fill_null(False)


def sort_frame(
    frame: pl.LazyFrame, state: TableState, config: TableConfig
) -> pl.LazyFrame:
    """Sort a frame by a state's sort column.

    Parameters
    ----------
    frame : pl.LazyFrame
        Rows in the frame's order.
    state : TableState
        State naming the sort column and direction.
    config : TableConfig
        Table settings.

    Returns
    -------
    pl.LazyFrame
        Rows sorted by the column, with null values last and ties in the
        frame's order. A ``frame_order`` column keeps the frame's order, or
        reverses it when descending; without a sort column the order is kept.
    """
    if state.sort_by is None:
        return frame
    if config.sortable_columns[state.sort_by].frame_order:
        return frame.reverse() if state.descending else frame
    return frame.sort(
        state.sort_by, descending=state.descending, nulls_last=True, maintain_order=True
    )


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


def apply_state(
    frame: pl.DataFrame | pl.LazyFrame,
    state: TableState,
    config: TableConfig,
    options: Mapping[str, Sequence[str]] | None = None,
) -> TableView:
    """Filter, sort, and page a frame for one table view.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table in its default order. It holds the key column,
        the shown columns, and any ``title_column``.
    state : TableState
        Sort order, filters, and page, for example from ``parse_state``.
    config : TableConfig
        Table settings.
    options : Mapping[str, Sequence[str]] | None, default None
        Values offered by each ``"choice"`` filter; ``choice_options`` of the
        frame if ``None``.

    Returns
    -------
    TableView
        The shown page and the keys of every matching row.
    """
    matching = sort_frame(frame.lazy().filter(filter_expression(state)), state, config).collect()
    page = paginate(matching.height, state.page, config.page_size)
    return TableView(
        state=state,
        page=page,
        rows=matching.slice(page.offset, page.length).to_dicts(),
        matching_keys=(
            matching.get_column(config.key).cast(pl.String).to_list() if config.selectable else []
        ),
        options=(
            choice_options(frame, config)
            if options is None
            else {name: list(values) for name, values in options.items()}
        ),
    )
