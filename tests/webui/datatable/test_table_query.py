"""Tests for filtering, sorting, and paging a polars frame as a data table."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from flat_pca.webui.datatable import (
    ColumnConfig,
    TableConfig,
    TableState,
    apply_state,
    choice_options,
    parse_state,
)


def _keys(frame: pl.DataFrame, state: TableState, config: TableConfig) -> list[str]:
    """Return the keys of every row matching a state, in display order."""
    return apply_state(frame, state, config).matching_keys


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        # Text: a case-insensitive substring.
        (TableState(text={"name": "ALPHA"}), ["k1", "k4"]),
        (TableState(text={"name": "a.p"}), []),
        # Category: one value; nulls never match.
        (TableState(equals={"group": "a"}), ["k1", "k4"]),
        # Integer bounds are inclusive; nulls never match.
        (TableState(ranges={"count": (2.0, None)}), ["k1", "k4"]),
        (TableState(ranges={"count": (None, 2.0)}), ["k3", "k4"]),
        # Float bounds.
        (TableState(ranges={"score": (0.5, 1.0)}), ["k1", "k4"]),
        # Datetime bounds.
        (TableState(ranges={"when": (datetime.fromisoformat("2026-02-01"), None)}), ["k2", "k3"]),
        (
            TableState(ranges={"when": (datetime.fromisoformat("2026-01-01"), datetime.fromisoformat("2026-02-01"))}),
            ["k1", "k3"],
        ),
        # Filters combine with AND.
        (TableState(text={"name": "alpha"}, equals={"group": "a"}, ranges={"count": (3.0, None)}), ["k1"]),
    ],
)
def test_apply_state_filters_every_column_type(
    frame: pl.DataFrame, config: TableConfig, state: TableState, expected: list[str]
) -> None:
    """Each filter kind selects the matching rows in the frame's order."""
    assert _keys(frame.lazy(), state, config) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (TableState(), ["k1", "k2", "k3", "k4"]),
        (TableState(sort_by="key"), ["k1", "k2", "k3", "k4"]),
        (TableState(sort_by="key", descending=True), ["k4", "k3", "k2", "k1"]),
        (TableState(sort_by="name"), ["k1", "k3", "k4", "k2"]),
        (TableState(sort_by="group"), ["k1", "k4", "k2", "k3"]),
        (TableState(sort_by="group", descending=True), ["k2", "k1", "k4", "k3"]),
        (TableState(sort_by="count"), ["k3", "k4", "k1", "k2"]),
        (TableState(sort_by="score", descending=True), ["k2", "k4", "k1", "k3"]),
        (TableState(sort_by="when"), ["k1", "k3", "k2", "k4"]),
    ],
)
def test_apply_state_sorts_with_nulls_last_and_stable_ties(
    frame: pl.DataFrame, config: TableConfig, state: TableState, expected: list[str]
) -> None:
    """Sorting puts nulls last and keeps the frame's order among ties.

    ``key`` is a ``frame_order`` column, so it keeps or reverses the frame.
    """
    assert _keys(frame, state, config) == expected


def test_apply_state_returns_one_page_and_every_matching_key(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """The rows are one page; the keys cover every page; a far page shows the last."""
    view = apply_state(frame, TableState(page=9), config)

    assert [row["key"] for row in view.rows] == ["k3", "k4"]
    assert view.rows[0]["note"] == "third"
    assert (view.page.number, view.page.count, view.page.start, view.page.end) == (2, 2, 3, 4)
    assert view.matching_keys == ["k1", "k2", "k3", "k4"]


def test_apply_state_without_matches_has_an_empty_page(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """No matching row gives one empty page."""
    view = apply_state(frame, TableState(text={"name": "zzz"}), config)

    assert view.rows == []
    assert (view.page.total, view.page.start, view.page.end) == (0, 0, 0)


def test_apply_state_lists_no_keys_for_a_table_without_selection(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """Only a selectable table lists the matching keys."""
    plain = TableConfig(table_id="t", key="key", columns=config.columns, url="/table")

    assert apply_state(frame, TableState(), plain).matching_keys == []


def test_choice_options_come_from_the_frame_or_the_caller(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """Choices are the sorted distinct values unless the caller gives them."""
    assert choice_options(frame, config) == {"group": ["a", "b"]}
    assert apply_state(frame, TableState(), config).options == {"group": ["a", "b"]}
    given = apply_state(frame, TableState(), config, {"group": ("b", "z")})
    assert given.options == {"group": ["b", "z"]}


def test_apply_state_keeps_integer_bounds_exact() -> None:
    """Integer bounds are not rounded to floats against integer columns."""
    frame = pl.DataFrame({"key": ["a", "b"], "big": [2**53, 2**53 + 1]})
    config = TableConfig(
        table_id="t",
        key="key",
        columns=(ColumnConfig("key"), ColumnConfig("big", filter="number")),
        url="/",
        selectable=True,
    )

    state = parse_state({"t.min__big": [str(2**53 + 1)]}, config)

    assert state.ranges == {"big": (2**53 + 1, None)}
    assert apply_state(frame, state, config).matching_keys == ["b"]


def test_apply_state_gives_the_same_text_for_row_and_matching_keys() -> None:
    """Keys of any type have one text form for the rows and the selection."""
    frame = pl.DataFrame({"when": [datetime.fromisoformat("2026-01-01")], "flag": [True]})
    for key in ("when", "flag"):
        config = TableConfig(
            table_id="t", key=key, columns=(ColumnConfig(key),), url="/", selectable=True
        )

        view = apply_state(frame.lazy(), TableState(), config)

        assert view.row_keys == view.matching_keys
        assert view.row_keys == frame.select(pl.col(key).cast(pl.String)).to_series().to_list()


def test_apply_state_pages_a_lazy_frame_without_selection(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """A table without selection counts the matching rows of a LazyFrame."""
    plain = TableConfig(
        table_id="t", key="key", columns=config.columns, url="/table", page_size=3
    )

    view = apply_state(frame.lazy(), TableState(sort_by="count", page=2), plain)

    assert view.page.total == 4
    assert view.row_keys == ["k2"]
    assert [row["key"] for row in view.rows] == ["k2"]


@pytest.mark.parametrize(
    ("values", "bound", "expected"),
    [
        # Integer bounds past the Polars integer range against a float column.
        ([1e38, 1e40], str(10**39), ["b"]),
        ([1e38, 1e40], str(10**400), []),
        # A float bound against an integer column.
        ([1, 3], "2.5", ["b"]),
        # An integer bound past 64 bits against an integer column.
        ([1, 3], str(2**70), []),
    ],
)
def test_apply_state_fits_bounds_to_the_column_type(
    values: list[float], bound: str, expected: list[str]
) -> None:
    """Bounds that parse are compared with the column without failing."""
    frame = pl.DataFrame({"key": ["a", "b"], "value": values})
    config = TableConfig(
        table_id="t",
        key="key",
        columns=(ColumnConfig("key"), ColumnConfig("value", filter="number")),
        url="/",
        selectable=True,
    )

    state = parse_state({"t.min__value": [bound]}, config)

    assert apply_state(frame, state, config).matching_keys == expected


def test_apply_state_shows_binary_keys_of_a_table_without_selection() -> None:
    """Only a selectable table casts its keys to text."""
    frame = pl.DataFrame({"key": [b"\xff", b"\xfe"], "label": ["a", "b"]})
    config = TableConfig(table_id="t", key="key", columns=(ColumnConfig("label"),), url="/")

    view = apply_state(frame, TableState(), config)

    assert [row["label"] for row in view.rows] == ["a", "b"]
    assert view.row_keys == [str(b"\xff"), str(b"\xfe")]


INT64_MIN = -(2**63)
UINT64_MAX = 2**64 - 1


@pytest.mark.parametrize(
    ("dtype", "values", "side", "bound", "expected"),
    [
        # Bounds just past the limits of the column type.
        (pl.Int64, [INT64_MIN, INT64_MIN + 1], "max", INT64_MIN - 1, []),
        (pl.Int64, [INT64_MIN, INT64_MIN + 1], "min", INT64_MIN - 1, ["a", "b"]),
        (pl.Int64, [INT64_MIN, INT64_MIN + 1], "max", INT64_MIN, ["a"]),
        (pl.UInt64, [UINT64_MAX - 1, UINT64_MAX], "min", UINT64_MAX + 1, []),
        (pl.UInt64, [UINT64_MAX - 1, UINT64_MAX], "max", UINT64_MAX + 1, ["a", "b"]),
        (pl.UInt64, [UINT64_MAX - 1, UINT64_MAX], "min", UINT64_MAX, ["b"]),
        (pl.Int64, [2**63 - 2, 2**63 - 1], "min", 2**63, []),
        (pl.Int64, [2**63 - 2, 2**63 - 1], "max", 2**63, ["a", "b"]),
        (pl.UInt64, [UINT64_MAX - 1, UINT64_MAX], "min", -1, ["a", "b"]),
        (pl.UInt64, [UINT64_MAX - 1, UINT64_MAX], "max", -1, []),
        (pl.Int8, [-128, 127], "min", 127, ["b"]),
        (pl.Int8, [-128, 127], "max", 200, ["a", "b"]),
        (pl.UInt8, [0, 255], "min", 256, []),
        # Bounds inside a 128-bit column.
        (pl.UInt128, [2**128 - 2, 2**128 - 1], "min", 2**128 - 1, ["b"]),
        (pl.UInt128, [2**128 - 2, 2**128 - 1], "min", 2**128, []),
        (pl.Int128, [2**70, 2**70 + 1], "min", 2**70 + 1, ["b"]),
        # Bounds past every integer type of polars.
        (pl.Int64, [1, 2], "min", 2**200, []),
        (pl.Int64, [1, 2], "max", 2**200, ["a", "b"]),
        (pl.Int64, [1, 2], "min", -(2**200), ["a", "b"]),
        (pl.Int64, [1, 2], "max", -(2**200), []),
    ],
)
def test_apply_state_compares_integer_columns_exactly(
    dtype: pl.DataType, values: list[int], side: str, bound: int, expected: list[str]
) -> None:
    """Integer bounds keep their precision against integer columns of any width."""
    frame = pl.DataFrame({"key": ["a", "b"], "value": pl.Series(values, dtype=dtype)})
    config = TableConfig(
        table_id="t",
        key="key",
        columns=(ColumnConfig("key"), ColumnConfig("value", filter="number")),
        url="/",
        selectable=True,
    )

    state = parse_state({f"t.{side}__value": [str(bound)]}, config)

    assert apply_state(frame, state, config).matching_keys == expected


def test_apply_state_out_of_range_bounds_skip_nulls() -> None:
    """A bound past every integer type still matches no null value."""
    frame = pl.DataFrame({"key": ["a", "b"], "value": [1, None]})
    config = TableConfig(
        table_id="t",
        key="key",
        columns=(ColumnConfig("key"), ColumnConfig("value", filter="number")),
        url="/",
        selectable=True,
    )

    state = parse_state({"t.min__value": [str(-(2**200))]}, config)

    assert apply_state(frame, state, config).matching_keys == ["a"]
