"""Applying a table state to a polars frame: filtering, sorting, and paging."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .chips import FilterChip, filter_chips
from .config import TableConfig
from .counts import ColumnCounts, count_values
from .formatting import dtype_label
from .pagination import Page, paginate
from .state import Bound, TableState

# The width of each polars integer type.
INTEGER_BITS = (
    (pl.Int8, 8),
    (pl.Int16, 16),
    (pl.Int32, 32),
    (pl.Int64, 64),
    (pl.Int128, 128),
    (pl.UInt8, 8),
    (pl.UInt16, 16),
    (pl.UInt32, 32),
    (pl.UInt64, 64),
    (pl.UInt128, 128),
)


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
    counts : ColumnCounts
        Rows of each value of the ``"choice"`` columns and null values of
        the filtered columns, over every row of the table.
    types : dict[str, str]
        Type of each shown column (``dtype_label``), keyed by column name.
    chips : list[FilterChip]
        Column filters in use (``filter_chips``).
    """

    state: TableState
    page: Page
    rows: list[dict[str, object]]
    row_keys: list[str]
    matching_keys: list[str]
    options: dict[str, list[str]]
    counts: ColumnCounts
    types: dict[str, str]
    chips: list[FilterChip]


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


def integer_range(dtype: pl.DataType) -> tuple[int, int]:
    """Return the smallest and largest value of an integer type.

    Parameters
    ----------
    dtype : pl.DataType
        Integer type, signed or unsigned, of 8 to 128 bits.

    Returns
    -------
    tuple[int, int]
        Inclusive range of the type.
    """
    bits = next(bits for kind, bits in INTEGER_BITS if dtype == kind)
    if dtype.is_signed_integer():
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


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
        is compared exactly with an integer column: inside the column type's
        range, as a literal of that type; outside it, every non-null value is
        on the inner side of the bound or none is. Against other columns an
        ``int`` bound is compared as a float (``±inf`` past the float range).
    """
    column = pl.col(name)
    if isinstance(bound, int):
        if dtype.is_integer():
            smallest, largest = integer_range(dtype)
            if smallest <= bound <= largest:
                literal = pl.lit(bound, dtype=dtype)
            else:
                # Every value is above a bound below the type, and below one
                # above it; null values match no filter.
                return column.is_not_null() if (bound < smallest) == lower else pl.lit(False)
        else:
            try:
                literal = pl.lit(float(bound))
            except OverflowError:
                literal = pl.lit(math.inf if bound > 0 else -math.inf)
    else:
        literal = pl.lit(bound)
    return column >= literal if lower else column <= literal


def search_terms(text: str) -> list[str]:
    """Split a search into the lowercase words a value must contain.

    Parameters
    ----------
    text : str
        Search typed into a ``"text"`` filter.

    Returns
    -------
    list[str]
        Words separated by whitespace, in lowercase, without repeats.
    """
    return list(dict.fromkeys(text.lower().split()))


def search_columns(config: TableConfig, schema: Mapping[str, pl.DataType]) -> list[str]:
    """Return the columns that the search of the whole table looks in.

    Parameters
    ----------
    config : TableConfig
        Table settings.
    schema : Mapping[str, pl.DataType]
        Column types of the frame.

    Returns
    -------
    list[str]
        Shown columns of type ``pl.String``, in display order.
    """
    return [column.name for column in config.columns if schema[column.name] == pl.String]


def _contains(name: str, term: str) -> pl.Expr:
    """Return whether a column's text contains a lowercase word, in any case."""
    return pl.col(name).cast(pl.String).str.to_lowercase().str.contains(term, literal=True)


def filter_expression(
    state: TableState, config: TableConfig, schema: Mapping[str, pl.DataType]
) -> pl.Expr:
    """Return the expression selecting the rows that match a state's filters.

    Parameters
    ----------
    state : TableState
        State whose column names have been validated, for example by
        ``parse_state``.
    config : TableConfig
        Table settings, which decide the columns of the search of the whole
        table (``search_columns``).
    schema : Mapping[str, pl.DataType]
        Column types of the filtered frame, which decide the type of the
        range bounds (``bound_condition``) and the searched columns.

    Returns
    -------
    pl.Expr
        Boolean expression combining every filter with AND. A column search
        matches a value containing each of its words (``search_terms``) as
        plain text, in any order and case; the search of the whole table
        matches a row whose ``search_columns`` hold each word, each in any of
        them (no row without such a column); a ``"choice"`` filter matches
        any of its values. Null values match no search, value, or bound, so
        those filters drop them; a null filter keeps only the null values
        (``"is_null"``) or drops them (``"is_not_null"``).
    """
    conditions = [pl.lit(True)]
    searched = search_columns(config, schema)
    for term in search_terms(state.search):
        conditions.append(
            pl.any_horizontal(_contains(name, term) for name in searched)
            if searched
            else pl.lit(False)
        )
    for name, text in state.text.items():
        conditions.extend(_contains(name, term) for term in search_terms(text))
    for name, values in state.equals.items():
        conditions.append(pl.col(name).cast(pl.String).is_in(values))
    for name, (lower, upper) in state.ranges.items():
        if lower is not None:
            conditions.append(bound_condition(name, lower, schema[name], lower=True))
        if upper is not None:
            conditions.append(bound_condition(name, upper, schema[name], lower=False))
    for name, null_filter in state.nulls.items():
        column = pl.col(name)
        conditions.append(column.is_null() if null_filter == "is_null" else column.is_not_null())
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
        The shown page, the keys of every matching row, and what the column
        menus show. Only the page's rows are collected with every column; the
        matching rows are counted, or only their keys are collected for a
        selectable table.
    """
    lazy = frame.lazy()
    schema = lazy.collect_schema()
    matching = sort_frame(lazy.filter(filter_expression(state, config, schema)), state, config)
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
        counts=count_values(frame, config),
        types={column.name: dtype_label(schema[column.name]) for column in config.columns},
        chips=filter_chips(state, config),
    )
