"""Tests for the chips of the filters in use of a data table."""

from __future__ import annotations

from datetime import datetime

from flat_pca.webui.datatable import (
    FilterChip,
    TableConfig,
    TableState,
    filter_chips,
    parse_state,
)


def test_filter_chips_list_every_filter_in_column_order(config: TableConfig) -> None:
    """Each filter in use is one chip naming the parameters that its button clears."""
    state = parse_state(
        {
            "t.null__when": ["is_not_null"],
            "t.min__score": ["0.5"],
            "t.max__score": ["2"],
            "t.eq__group": ["b", "a"],
            "t.null__group": ["is_null"],
            "t.max__count": ["3"],
            "t.min__when": ["2026-01-01T12:30"],
            "t.q__name": [" alp  b "],
        },
        config,
    )

    assert filter_chips(state, config) == [
        FilterChip('name ~ "alp  b"', ("t.q__name",)),
        FilterChip("group ∈ {b, a}", ("t.eq__group",)),
        FilterChip("group is_null", ("t.null__group",)),
        FilterChip("count ≤ 3", ("t.min__count", "t.max__count")),
        FilterChip("0.5 ≤ score ≤ 2", ("t.min__score", "t.max__score")),
        FilterChip("when ≥ 2026-01-01 12:30:00", ("t.min__when", "t.max__when")),
        FilterChip("when is_not_null", ("t.null__when",)),
    ]


def test_filter_chips_name_columns_by_header_and_skip_the_search(config: TableConfig) -> None:
    """Chips use the column's label; the search of the whole table has no chip."""
    state = TableState(
        search="x",
        ranges={"count": (datetime.fromisoformat("2026-01-01"), None)},
        nulls={"key": "is_not_null"},
    )

    assert [chip.label for chip in filter_chips(state, config)] == [
        "Key is_not_null",
        "count ≥ 2026-01-01 00:00:00",
    ]
    assert filter_chips(TableState(search="x"), config) == []


def test_filter_chips_keep_the_precision_of_the_bounds(config: TableConfig) -> None:
    """Close float bounds and fractional seconds are shown as they filter."""
    state = parse_state(
        {
            "t.min__score": ["1.000001"],
            "t.max__score": ["1.000002"],
            "t.min__count": ["12345678901234567890"],
            "t.min__when": ["2026-01-01T12:30:00.250"],
        },
        config,
    )

    assert [chip.label for chip in filter_chips(state, config)] == [
        "count ≥ 12345678901234567890",
        "1.000001 ≤ score ≤ 1.000002",
        "when ≥ 2026-01-01 12:30:00.250000",
    ]
