"""Tests for filtering, sorting, and paging a polars frame as a data table."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from flat_pca.webui.datatable import (
    TableConfig,
    TableState,
    apply_state,
    choice_options,
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
