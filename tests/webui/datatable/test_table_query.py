"""Tests for filtering, sorting, and paging a polars frame as a data table."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import polars as pl
import pytest

from flat_pca.webui.datatable import (
    ColumnConfig,
    ColumnCounts,
    TableConfig,
    TableState,
    apply_state,
    choice_options,
    count_values,
    filter_expression,
    parse_state,
    search_terms,
    sort_frame,
)


def _keys(frame: pl.DataFrame, state: TableState, config: TableConfig) -> list[str]:
    """Return the keys of every row matching a state, in display order."""
    return apply_state(frame, state, config).matching_keys


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        # Text: every word as a case-insensitive substring, in any order.
        (TableState(text={"name": "ALPHA"}), ["k1", "k4"]),
        (TableState(text={"name": "BET alp"}), ["k4"]),
        (TableState(text={"name": "a.p"}), []),
        # Category: any of the values; nulls never match.
        (TableState(equals={"group": ("a",)}), ["k1", "k4"]),
        (TableState(equals={"group": ("b", "a")}), ["k1", "k2", "k4"]),
        (TableState(equals={"group": ("b", "zzz")}), ["k2"]),
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
        # Null filters of every column type.
        (TableState(nulls={"group": "is_null"}), ["k3"]),
        (TableState(nulls={"group": "is_not_null"}), ["k1", "k2", "k4"]),
        (TableState(nulls={"count": "is_null"}), ["k2"]),
        (TableState(nulls={"score": "is_not_null"}), ["k1", "k2", "k4"]),
        (TableState(nulls={"when": "is_null"}), ["k4"]),
        (TableState(nulls={"name": "is_null"}), []),
        # A value filter drops the nulls, so with "is_null" nothing matches.
        (TableState(equals={"group": ("a",)}, nulls={"group": "is_null"}), []),
        (TableState(ranges={"count": (2, None)}, nulls={"count": "is_not_null"}), ["k1", "k4"]),
        # Filters combine with AND.
        (TableState(text={"name": "alpha"}, equals={"group": ("a",)}, ranges={"count": (3.0, None)}), ["k1"]),
        (TableState(text={"name": "a"}, nulls={"when": "is_not_null"}), ["k1", "k2", "k3"]),
    ],
)
def test_apply_state_filters_every_column_type(
    frame: pl.DataFrame, config: TableConfig, state: TableState, expected: list[str]
) -> None:
    """Each filter kind selects the matching rows in the frame's order."""
    assert _keys(frame.lazy(), state, config) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        ("b01 lotA", ["LotA_B01_run3"]),
        ("RUN3", ["LotA_B01_run3", "lotb.b02(run3)"]),
        (".b02(", ["lotb.b02(run3)"]),
        ("(run3) lotb.", ["lotb.b02(run3)"]),
        ("[a-z]", []),
        ("a.b", []),
    ],
)
def test_text_filter_searches_words_as_plain_text(search: str, expected: list[str]) -> None:
    """A search's words match in any order and case and are not regular expressions."""
    frame = pl.DataFrame({"name": ["LotA_B01_run3", "lotb.b02(run3)", "axb"]})
    config = TableConfig(
        table_id="t",
        key="name",
        columns=(ColumnConfig("name", filter="text"),),
        url="/",
        selectable=True,
    )

    state = parse_state({"t.q__name": [search]}, config)

    assert apply_state(frame, state, config).matching_keys == expected


def test_search_terms_split_on_whitespace_in_lowercase() -> None:
    """Words are lowercased, split on any whitespace, and kept once."""
    assert search_terms("  B01\tlotA  b01 ") == ["b01", "lota"]
    assert search_terms("   ") == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"t.q__name": ["a"]},
        {"t.eq__group": ["a", "b"]},
        {"t.min__score": ["1"]},
        {"t.max__when": ["2026-02-01T00:00"]},
        {"t.null__count": ["is_not_null"]},
        {"t.null__group": ["is_null"]},
        {"t.eq__group": ["a"], "t.null__when": ["is_not_null"], "t.sort": ["score"]},
    ],
)
def test_matching_keys_follow_every_filter_over_all_pages(
    frame: pl.DataFrame, config: TableConfig, parameters: dict[str, list[str]]
) -> None:
    """The header checkbox's keys are exactly the filtered rows, shown on this page or not."""
    state = parse_state(parameters, config)
    expected = (
        frame.filter(filter_expression(state, frame.schema))
        .pipe(lambda rows: sort_frame(rows.lazy(), state, config).collect())
        .get_column("key")
        .to_list()
    )

    first = apply_state(frame, state, config)
    last = apply_state(frame, replace(state, page=first.page.count), config)

    assert first.matching_keys == expected
    assert last.matching_keys == expected
    shown = [*first.row_keys, *(last.row_keys if first.page.count > 1 else [])]
    assert shown == expected


def test_count_values_counts_choices_and_nulls_over_every_row(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """Choice values and nulls are counted over every row, whatever the filters."""
    counts = count_values(frame.lazy(), config)

    assert counts.values == {"group": {"a": 2, "b": 1}}
    assert counts.nulls == {"name": 0, "group": 1, "count": 1, "score": 1, "when": 1}
    filtered = apply_state(frame, TableState(equals={"group": ("b",)}), config)
    assert filtered.counts == counts


def test_count_values_keeps_column_names_apart_from_its_own() -> None:
    """Columns named like the count's fields are counted correctly."""
    frame = pl.DataFrame({"value": ["x", "x", None], "n": ["y", None, None]})
    config = TableConfig(
        table_id="t",
        key="value",
        columns=(ColumnConfig("value", filter="choice"), ColumnConfig("n", filter="choice")),
        url="/",
    )

    counts = count_values(frame, config)

    assert counts.values == {"value": {"x": 2}, "n": {"y": 1}}
    assert counts.nulls == {"value": 1, "n": 2}


def test_count_values_without_filters_is_empty(frame: pl.DataFrame) -> None:
    """A table without filters counts nothing."""
    config = TableConfig(table_id="t", key="key", columns=(ColumnConfig("key"),), url="/")

    assert count_values(frame, config) == ColumnCounts(values={}, nulls={})


def test_apply_state_names_the_type_of_every_shown_column(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """Each shown column has its short polars type name."""
    view = apply_state(frame, TableState(), config)

    assert view.types == {
        "key": "str",
        "name": "str",
        "group": "str",
        "count": "i64",
        "score": "f64",
        "when": "datetime[μs]",
        "note": "str",
    }


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
