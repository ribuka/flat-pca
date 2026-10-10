"""Tests for what a data table counts over every row of its frame."""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from flat_pca.webui.datatable import (
    TableConfig,
    TableState,
    apply_state,
    column_histograms,
    count_histograms,
    mark_histograms,
    parse_state,
    summarize_table,
)


def test_summary_holds_the_types_choices_counts_and_histograms(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """The summary counts every row once, whatever the state."""
    config = replace(config, histograms=True)

    summary = summarize_table(frame, config)

    assert summary.schema == dict(frame.schema)
    assert summary.types["count"] == "i64"
    assert summary.options == {"group": ["a", "b"]}
    assert summary.counts.values == {"group": {"a": 2, "b": 1}}
    assert summary.counts.nulls["count"] == 1
    assert list(summary.histograms) == ["count", "score", "when"]
    assert all(b.in_filter for b in summary.histograms["count"].bins)


def test_apply_state_takes_the_menus_from_a_given_summary(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """A kept summary is used as it is, so the frame is not counted again."""
    config = replace(config, histograms=True)
    summary = summarize_table(frame, config, {"group": ("a", "b", "z")})
    state = parse_state({"t.min__count": ["3"]}, config)

    view = apply_state(frame.head(1), state, config, summary)

    assert view.page.total == 1
    assert view.options == {"group": ["a", "b", "z"]}
    assert view.counts == summary.counts
    assert view.types == summary.types
    assert view.histograms == column_histograms(frame, state, config)


def test_marking_counted_histograms_equals_counting_with_the_state(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """Counting once and marking each state gives the histograms of that state."""
    state = parse_state(
        {"t.min__count": ["2"], "t.max__score": ["1"], "t.eq__group": ["a"]}, config
    )
    counted = count_histograms(frame, config)

    marked = mark_histograms(counted, state, frame.schema)

    assert marked == column_histograms(frame, state, config)
    assert [b.in_filter for b in marked["count"].bins] == [False, True, True]
    assert mark_histograms(counted, TableState(), frame.schema) == counted


def test_apply_state_pages_by_the_chosen_page_size(
    frame: pl.DataFrame, config: TableConfig
) -> None:
    """A page size in the state replaces the table's."""
    config = replace(config, page_sizes=(3,))

    default = apply_state(frame, TableState(), config)
    chosen = apply_state(frame, parse_state({"t.page_size": ["3"]}, config), config)

    assert (default.page_size, default.page.count, len(default.rows)) == (2, 2, 2)
    assert (chosen.page_size, chosen.page.count, len(chosen.rows)) == (3, 2, 3)
