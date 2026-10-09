"""Applying a table state to a polars frame: filtering, sorting, and paging."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .config import TableConfig
from .pagination import Page, paginate
from .state import Bound, TableState


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
    row_keys : list[str]
        Key of each row in ``rows`` as text: ``key_text`` for a selectable
        table, so that it matches the selection, and ``str`` of the value
        otherwise, so that any key type (binary, for example) can be shown.
    matching_keys : list[str]
        Keys of every row matching the filters, on all pages, as text; empty
        for a table that is not selectable.
    options : dict[str, list[str]]
        Values offered by each ``"choice"`` filter.
    """

    state: TableState
    page: Page
    rows: list[dict[str, object]]
    row_keys: list[str]
    matching_keys: list[str]
    options: dict[str, list[str]]


def key_text(config: TableConfig) -> pl.Expr:
    """Return the expression giving the text of each row's key.

    The row checkboxes, the header checkbox, and the selection all use this
    text, so that keys of any type (datetimes, booleans, ...) match.

    Parameters
    ----------
    config : TableConfig
        Table settings naming the key column.

    Returns
    -------
    pl.Expr
        The key column cast to ``pl.String``.
    """
    return pl.col(config.key).cast(pl.String)


def bound_condition(name: str, bound: Bound, dtype: pl.DataType, *, lower: bool) -> pl.Expr:
    """Return the condition that a column is on the inner side of a bound.

    Parameters
    ----------
    name : str
        Column name.
    bound : Bound
        Bound from ``parse_state``; not ``None``.
    dtype : pl.DataType
        Type of the column.
    lower : bool
        Whether the bound is the lower one (``>=``) or the upper one (``<=``).

    Returns
    -------
    pl.Expr
        Inclusive comparison of the column with the bound. An ``int`` bound
        is compared exactly with an integer column; one past every integer
        type of polars is outside any integer column, so every non-null value is
        on the inner side of it or none is. Against other columns an ``int`` bound
        is compared as a float (``±inf`` past the float range).
    """
    column = pl.col(name)
    if isinstance(bound, int):
        if dtype.is_integer():
            try:
                literal = pl.lit(bound)
            except (OverflowError, pl.exceptions.InvalidOperationError):
                # Every value is above a bound below every integer type, and
                # below one above them; null values match no filter.
                return column.is_not_null() if (bound < 0) == lower else pl.lit(False)
        else:
            try:
                literal = pl.lit(float(bound))
            except OverflowError:
                literal = pl.lit(math.inf if bound > 0 else -math.inf)
    else:
        literal = pl.lit(bound)
    return column >= literal if lower else column <= literal


def filter_expression(state: TableState, schema: Mapping[str, pl.DataType]) -> pl.Expr:
    """Return the expression selecting the rows that match a state's filters.

    Parameters
    ----------
    state : TableState
        State whose column names have been validated, for example by
        ``parse_state``.
    schema : Mapping[str, pl.DataType]
        Column types of the filtered frame, which decide the type of the
        range bounds (``bound_condition``).

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
            conditions.append(bound_condition(name, lower, schema[name], lower=True))
        if upper is not None:
            conditions.append(bound_condition(name, upper, schema[name], lower=False))
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
        The shown page and the keys of every matching row. Only the page's
        rows are collected with every column; the matching rows are counted,
        or only their keys are collected for a selectable table.
    """
    lazy = frame.lazy()
    matching = sort_frame(
        lazy.filter(filter_expression(state, lazy.collect_schema())), state, config
    )
    if config.selectable:
        matching_keys = matching.select(key_text(config)).collect().to_series().to_list()
        total = len(matching_keys)
    else:
        matching_keys = []
        total = matching.select(pl.len()).collect().item()
    page = paginate(total, state.page, config.page_size)
    rows = matching.slice(page.offset, page.length).collect()
    return TableView(
        state=state,
        page=page,
        rows=rows.to_dicts(),
        row_keys=(
            rows.select(key_text(config)).to_series().to_list()
            if config.selectable
            else [str(value) for value in rows.get_column(config.key).to_list()]
        ),
        matching_keys=matching_keys,
        options=(
            choice_options(frame, config)
            if options is None
            else {name: list(values) for name, values in options.items()}
        ),
    )
