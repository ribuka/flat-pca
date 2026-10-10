"""Applying a table state to a polars frame: filtering, sorting, and paging."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from .bounds import bound_condition
from .chips import FilterChip, filter_chips
from .config import TableConfig
from .counts import ColumnCounts
from .histograms import ColumnHistogram, mark_histograms
from .pagination import Page, paginate
from .state import TableState
from .summary import TableSummary, summarize_table


@dataclass(frozen=True)
class TableView:
    """What one page of a data table shows.

    Attributes
    ----------
    state : TableState
        State the view was made for.
    page : Page
        Position of the shown page; a page past the last one shows the last.
    page_size : int
        Rows per page: the state's ``page_size``, or the table's.
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
    histograms : dict[str, ColumnHistogram]
        Distribution of each shown column over every row with the state's
        filters marked (``mark_histograms``); empty unless
        ``TableConfig.histograms``.
    """

    state: TableState
    page: Page
    page_size: int
    rows: list[dict[str, object]]
    row_keys: list[str]
    matching_keys: list[str]
    options: dict[str, list[str]]
    counts: ColumnCounts
    types: dict[str, str]
    chips: list[FilterChip]
    histograms: dict[str, ColumnHistogram]


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


def matching_rows(
    frame: pl.LazyFrame,
    state: TableState,
    config: TableConfig,
    schema: Mapping[str, pl.DataType] | None = None,
) -> pl.LazyFrame:
    """Return the rows matching a state's filters, in its sort order.

    The table view (``apply_state``), the header checkbox's matching keys,
    and the exported rows (``export_rows``) all take their rows from here.

    Parameters
    ----------
    frame : pl.LazyFrame
        Every row of the table in its default order.
    state : TableState
        Sort order and filters.
    config : TableConfig
        Table settings.
    schema : Mapping[str, pl.DataType] | None, default None
        Column types of the frame; collected from it if ``None``.

    Returns
    -------
    pl.LazyFrame
        The rows that ``filter_expression`` keeps, sorted by ``sort_frame``.
    """
    if schema is None:
        schema = frame.collect_schema()
    return sort_frame(frame.filter(filter_expression(state, config, schema)), state, config)


def apply_state(
    frame: pl.DataFrame | pl.LazyFrame,
    state: TableState,
    config: TableConfig,
    summary: TableSummary | None = None,
) -> TableView:
    """Filter, sort, and page a frame for one table view.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table in its default order. It holds the key column,
        the shown columns, and any ``title_column``.
    state : TableState
        Sort order, filters, page, and page size, for example from
        ``parse_state``.
    config : TableConfig
        Table settings.
    summary : TableSummary | None, default None
        ``summarize_table`` of the same frame and settings, which an
        application can keep while the frame does not change; made from the
        frame if ``None``.

    Returns
    -------
    TableView
        The shown page, the keys of every matching row, and what the column
        menus show. Only the page's rows are collected with every column; the
        matching rows are counted, or only their keys are collected for a
        selectable table. The counts and histograms are the summary's, over
        every row of the frame.
    """
    if summary is None:
        summary = summarize_table(frame, config)
    lazy = frame.lazy()
    matching = matching_rows(lazy, state, config, summary.schema)
    if config.selectable:
        matching_keys = matching.select(key_text(config)).collect().to_series().to_list()
        total = len(matching_keys)
    else:
        matching_keys = []
        total = matching.select(pl.len()).collect().item()
    page_size = state.page_size or config.page_size
    page = paginate(total, state.page, page_size)
    rows = matching.slice(page.offset, page.length).collect()
    return TableView(
        state=state,
        page=page,
        page_size=page_size,
        rows=rows.to_dicts(),
        row_keys=(
            rows.select(key_text(config)).to_series().to_list()
            if config.selectable
            else [str(value) for value in rows.get_column(config.key).to_list()]
        ),
        matching_keys=matching_keys,
        options=summary.options,
        counts=summary.counts,
        types=summary.types,
        chips=filter_chips(state, config),
        histograms=mark_histograms(summary.histograms, state, summary.schema),
    )
