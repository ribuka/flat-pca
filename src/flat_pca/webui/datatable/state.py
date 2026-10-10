"""The state of a data table (sort, filters, page) and its query parameters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, get_args

from .config import ColumnConfig, TableConfig

SORT_PARAMETER = "sort"
ORDER_PARAMETER = "order"
PAGE_PARAMETER = "page"
TEXT_PREFIX = "q__"
EQUALS_PREFIX = "eq__"
MIN_PREFIX = "min__"
MAX_PREFIX = "max__"
NULL_PREFIX = "null__"
# The filter parameter prefixes and the filter kinds they fit.
FILTER_PREFIXES = {
    TEXT_PREFIX: ("text",),
    EQUALS_PREFIX: ("choice",),
    MIN_PREFIX: ("number", "datetime"),
    MAX_PREFIX: ("number", "datetime"),
    NULL_PREFIX: ("text", "choice", "number", "datetime"),
}

# How a null filter treats null values: keep only them, or drop them.
type NullFilter = Literal["is_null", "is_not_null"]
NULL_FILTERS: tuple[NullFilter, ...] = get_args(NullFilter.__value__)

# A "number" bound kept as an int instead of a float.
INTEGER = re.compile(r"[+-]?[0-9]+")

type Bound = int | float | datetime | None


@dataclass(frozen=True)
class TableState:
    """Sort order, filters, and page of a data table.

    Attributes
    ----------
    sort_by : str | None, default None
        Column to sort by, or ``None`` for the frame's order.
    descending : bool, default False
        Whether to sort in descending order.
    page : int, default 1
        1-based number of the shown page.
    text : dict[str, str]
        Searches of ``"text"`` columns: words separated by whitespace that a
        value must all contain, in any order and case.
    equals : dict[str, tuple[str, ...]]
        Values of ``"choice"`` columns, one of which a row must have.
    ranges : dict[str, tuple[Bound, Bound]]
        Inclusive ``(lower, upper)`` bounds of ``"number"`` and
        ``"datetime"`` columns; ``None`` leaves a side open.
    nulls : dict[str, NullFilter]
        Columns whose null values are the only ones kept (``"is_null"``) or
        are dropped (``"is_not_null"``).
    """

    sort_by: str | None = None
    descending: bool = False
    page: int = 1
    text: dict[str, str] = field(default_factory=dict)
    equals: dict[str, tuple[str, ...]] = field(default_factory=dict)
    ranges: dict[str, tuple[Bound, Bound]] = field(default_factory=dict)
    nulls: dict[str, NullFilter] = field(default_factory=dict)

    def is_filtered(self, name: str) -> bool:
        """Return whether any filter of the state uses a column.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        bool
            Whether the column has a search, values, bounds, or a null filter.
        """
        return any(name in filters for filters in (self.text, self.equals, self.ranges, self.nulls))


def parameter_name(config: TableConfig, name: str) -> str:
    """Return the query parameter name of one table control.

    Parameters
    ----------
    config : TableConfig
        Table settings.
    name : str
        Control name, such as ``"sort"`` or ``"min__<column>"``.

    Returns
    -------
    str
        ``"<table_id>.<name>"``.
    """
    return config.prefix + name


def _parse_bound(raw: str, column: ColumnConfig, key: str) -> Bound:
    """Parse one range bound from a query parameter.

    Parameters
    ----------
    raw : str
        Parameter value; blank means an open bound.
    column : ColumnConfig
        Column settings deciding the value type.
    key : str
        Parameter name used in errors.

    Returns
    -------
    Bound
        Parsed bound, or ``None`` for a blank value. A ``"number"`` bound
        written as an integer stays an ``int`` so that it keeps its precision
        against integer columns.

    Raises
    ------
    ValueError
        If the value cannot be parsed.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        if column.filter == "datetime":
            return datetime.fromisoformat(text)
        return int(text) if INTEGER.fullmatch(text) else float(text)
    except ValueError as error:
        raise ValueError(f"invalid bound for {key!r}: {raw!r}") from error


def _last(values: Sequence[str]) -> str:
    """Return the last value of a query parameter, or ``""`` without one."""
    return values[-1] if values else ""


def _parse_null_filter(raw: str, key: str) -> NullFilter | None:
    """Parse a null filter from a query parameter.

    Parameters
    ----------
    raw : str
        Parameter value; blank means no null filter.
    key : str
        Parameter name used in errors.

    Returns
    -------
    NullFilter | None
        ``"is_null"`` or ``"is_not_null"``, or ``None`` for a blank value.

    Raises
    ------
    ValueError
        If the value is neither.
    """
    text = raw.strip()
    if not text:
        return None
    for value in NULL_FILTERS:
        if text == value:
            return value
    raise ValueError(f"invalid null filter for {key!r}: {raw!r}")


def parse_state(parameters: Mapping[str, Sequence[str]], config: TableConfig) -> TableState:
    """Build a ``TableState`` from query parameters.

    Only parameters named ``<table_id>.<name>`` belong to the table; others
    are ignored, so several tables can share one query string. The names are
    ``sort``, ``order`` (``asc`` or ``desc``), ``page`` (a positive integer),
    and, per filtered column, ``q__<column>`` (``"text"``), ``eq__<column>``
    (``"choice"``; repeated once per value), ``min__<column>`` /
    ``max__<column>`` (``"number"`` and ``"datetime"``), and
    ``null__<column>`` (``is_null`` or ``is_not_null``; every filter kind).
    Another repeated parameter uses its last value; blank values are ignored.

    Parameters
    ----------
    parameters : Mapping[str, Sequence[str]]
        Query parameters with every value of each name.
    config : TableConfig
        Table settings.

    Returns
    -------
    TableState
        Parsed state.

    Raises
    ------
    ValueError
        If a parameter of the table is unknown, names a column without that
        filter, or has an unparsable value, sort column, order, or page.
    """
    every = {
        key.removeprefix(config.prefix): raw
        for key, raw in parameters.items()
        if key.startswith(config.prefix)
    }
    values = {key: _last(raw) for key, raw in every.items()}
    filtered = config.filtered_columns
    text: dict[str, str] = {}
    equals: dict[str, tuple[str, ...]] = {}
    lower: dict[str, Bound] = {}
    upper: dict[str, Bound] = {}
    nulls: dict[str, NullFilter] = {}
    for key, value in values.items():
        if key in (SORT_PARAMETER, ORDER_PARAMETER, PAGE_PARAMETER):
            continue
        prefix = next((prefix for prefix in FILTER_PREFIXES if key.startswith(prefix)), None)
        if prefix is None:
            raise ValueError(f"unknown parameter: {config.prefix + key!r}")
        name = key.removeprefix(prefix)
        column = filtered.get(name)
        if column is None:
            raise ValueError(f"unknown filter column: {name!r}")
        if column.filter not in FILTER_PREFIXES[prefix]:
            raise ValueError(f"filter {config.prefix + key!r} does not fit column type")
        if prefix == TEXT_PREFIX:
            if value.strip():
                text[name] = value.strip()
        elif prefix == EQUALS_PREFIX:
            chosen = tuple(dict.fromkeys(raw.strip() for raw in every[key] if raw.strip()))
            if chosen:
                equals[name] = chosen
        elif prefix == NULL_PREFIX:
            null_filter = _parse_null_filter(value, config.prefix + key)
            if null_filter is not None:
                nulls[name] = null_filter
        else:
            bound = _parse_bound(value, column, config.prefix + key)
            (lower if prefix == MIN_PREFIX else upper)[name] = bound

    sort_by = values.get(SORT_PARAMETER, "") or config.default_sort
    if sort_by is not None and sort_by not in config.sortable_columns:
        raise ValueError(f"unknown sort column: {sort_by!r}")
    order = values.get(ORDER_PARAMETER, "") or "asc"
    if order not in ("asc", "desc"):
        raise ValueError(f"invalid sort order: {order!r}")
    page = values.get(PAGE_PARAMETER, "").strip() or "1"
    if not (page.isascii() and page.isdigit()) or int(page) < 1:
        raise ValueError(f"invalid page: {page!r}")

    ranges = {
        name: (lower.get(name), upper.get(name))
        for name in dict.fromkeys([*lower, *upper])
        if lower.get(name) is not None or upper.get(name) is not None
    }
    return TableState(
        sort_by=sort_by,
        descending=order == "desc",
        page=int(page),
        text=text,
        equals=equals,
        ranges=ranges,
        nulls=nulls,
    )
