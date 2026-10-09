"""The state of a data table (sort, filters, page) and its query parameters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .config import ColumnConfig, TableConfig

SORT_PARAMETER = "sort"
ORDER_PARAMETER = "order"
PAGE_PARAMETER = "page"
TEXT_PREFIX = "q__"
EQUALS_PREFIX = "eq__"
MIN_PREFIX = "min__"
MAX_PREFIX = "max__"
# The filter parameter prefixes and the filter kinds they fit.
FILTER_PREFIXES = {
    TEXT_PREFIX: ("text",),
    EQUALS_PREFIX: ("choice",),
    MIN_PREFIX: ("number", "datetime"),
    MAX_PREFIX: ("number", "datetime"),
}

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
        Case-insensitive substrings that ``"text"`` columns must contain.
    equals : dict[str, str]
        Required values of ``"choice"`` columns.
    ranges : dict[str, tuple[Bound, Bound]]
        Inclusive ``(lower, upper)`` bounds of ``"number"`` and
        ``"datetime"`` columns; ``None`` leaves a side open.
    """

    sort_by: str | None = None
    descending: bool = False
    page: int = 1
    text: dict[str, str] = field(default_factory=dict)
    equals: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[Bound, Bound]] = field(default_factory=dict)


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


def parse_state(parameters: Mapping[str, Sequence[str]], config: TableConfig) -> TableState:
    """Build a ``TableState`` from query parameters.

    Only parameters named ``<table_id>.<name>`` belong to the table; others
    are ignored, so several tables can share one query string. The names are
    ``sort``, ``order`` (``asc`` or ``desc``), ``page`` (a positive integer),
    and, per filtered column, ``q__<column>`` (``"text"``), ``eq__<column>``
    (``"choice"``), or ``min__<column>`` / ``max__<column>`` (``"number"``
    and ``"datetime"``). A repeated parameter uses its last value; blank
    values are ignored.

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
    values = {
        key.removeprefix(config.prefix): _last(raw)
        for key, raw in parameters.items()
        if key.startswith(config.prefix)
    }
    filtered = config.filtered_columns
    text: dict[str, str] = {}
    equals: dict[str, str] = {}
    lower: dict[str, Bound] = {}
    upper: dict[str, Bound] = {}
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
        if prefix in (TEXT_PREFIX, EQUALS_PREFIX):
            if value.strip():
                (text if prefix == TEXT_PREFIX else equals)[name] = value.strip()
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
    )
