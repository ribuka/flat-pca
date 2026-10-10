"""Tests for the distributions of the data table's columns under their names."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from itertools import pairwise

import polars as pl
import pytest

from flat_pca.webui.datatable import (
    ColumnConfig,
    Histogram,
    HistogramBin,
    TableConfig,
    TableState,
    TopValue,
    TopValues,
    apply_state,
    column_histograms,
    float_bins,
    integer_bins,
    parse_state,
)


def _config(*columns: ColumnConfig) -> TableConfig:
    """Return a table of the given columns keyed by ``key``."""
    return TableConfig(table_id="t", key="key", columns=columns, url="/t", histograms=True)


def _bins(histogram: object) -> list[tuple[str, str, int, bool]]:
    """Return the bins of a histogram as tuples."""
    assert isinstance(histogram, Histogram)
    return [(b.lower, b.upper, b.count, b.in_filter) for b in histogram.bins]


def test_integer_bins_hold_one_value_each_in_a_short_range() -> None:
    """A range of at most 20 values has one bin per value."""
    assert integer_bins(3, 3) == [(3, 3)]
    assert integer_bins(1, 5) == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert len(integer_bins(-10, 9)) == 20


def test_integer_bins_of_a_long_range_have_equal_width() -> None:
    """Longer ranges get at most 20 bins of equal width, the last ending at the maximum."""
    assert integer_bins(0, 20) == [(first, first + 1) for first in range(0, 20, 2)] + [(20, 20)]
    bins = integer_bins(0, 999)
    assert len(bins) == 20
    assert bins[0] == (0, 49)
    assert bins[-1] == (950, 999)
    bins = integer_bins(-(2**63), 2**63 - 1)
    assert len(bins) == 20
    assert bins[0][0] == -(2**63)
    assert bins[-1][1] == 2**63 - 1
    assert all(last + 1 == first for (_, last), (first, _) in pairwise(bins))


def test_float_bins_cut_the_range_into_twenty() -> None:
    """Floats get 20 bins of equal width from the minimum to the maximum."""
    bins = float_bins(0.0, 2.0)
    assert len(bins) == 20
    assert bins[0] == (0.0, 0.1)
    assert bins[-1][1] == 2.0
    assert all(upper == lower for (_, upper), (lower, _) in pairwise(bins))
    assert float_bins(1.5, 1.5) == [(1.5, 1.5)]


def test_integer_histogram_counts_every_row_and_ignores_nulls() -> None:
    """Each bin counts the non-null values in it."""
    frame = pl.DataFrame({"key": list("abcde"), "n": [1, 2, 2, 5, None]})
    config = _config(ColumnConfig("key"), ColumnConfig("n", filter="number"))

    histograms = column_histograms(frame, TableState(), config)

    assert list(histograms) == ["n"]
    assert _bins(histograms["n"]) == [
        ("1", "1", 1, True),
        ("2", "2", 2, True),
        ("3", "3", 0, True),
        ("4", "4", 0, True),
        ("5", "5", 1, True),
    ]
    assert (histograms["n"].minimum, histograms["n"].maximum) == ("1", "5")


def test_float_histogram_skips_nan_and_infinite_values() -> None:
    """Float bins are half-open but the last; NaN and ±inf are not counted."""
    frame = pl.DataFrame(
        {"key": list("abcdef"), "x": [0.0, 0.05, 0.5, 1.0, float("nan"), float("inf")]}
    )
    config = _config(ColumnConfig("key"), ColumnConfig("x"))

    histogram = column_histograms(frame, TableState(), config)["x"]

    bins = _bins(histogram)
    assert len(bins) == 20
    assert bins[0] == ("0", "0.05", 1, True)
    assert bins[1] == ("0.05", "0.1", 1, True)
    assert bins[10] == ("0.5", "0.55", 1, True)
    assert bins[-1] == ("0.95", "1", 1, True)
    assert sum(count for _, _, count, _ in bins) == 4
    assert (histogram.minimum, histogram.maximum) == ("0", "1")


def test_single_value_columns_have_one_bin() -> None:
    """A float column with one distinct value has one bin holding every row."""
    frame = pl.DataFrame({"key": list("ab"), "x": [2.5, 2.5]})

    histogram = column_histograms(frame, TableState(), _config(ColumnConfig("x")))["x"]

    assert _bins(histogram) == [("2.5", "2.5", 2, True)]


def test_temporal_histograms_bin_by_time() -> None:
    """Datetimes and dates are binned on their time and written in their own format."""
    frame = pl.DataFrame(
        {
            "key": list("abc"),
            "when": [
                datetime.fromisoformat(f"2026-01-01T00:00:{second:02}") for second in (0, 10, 19)
            ],
            "day": [date(2026, 1, 1), date(2026, 1, 3), None],
        },
        schema_overrides={"when": pl.Datetime("ms")},
    )
    config = _config(ColumnConfig("when"), ColumnConfig("day"))

    histograms = column_histograms(frame, TableState(), config)

    when = _bins(histograms["when"])
    assert len(when) == 20
    assert when[0] == ("2026-01-01 00:00:00", "2026-01-01 00:00:00", 1, True)
    assert when[10][2] == 1
    assert when[-1][2] == 1
    assert histograms["when"].maximum == "2026-01-01 00:00:19"
    assert _bins(histograms["day"]) == [
        ("2026-01-01", "2026-01-01", 1, True),
        ("2026-01-02", "2026-01-02", 0, True),
        ("2026-01-03", "2026-01-03", 1, True),
    ]


def test_histograms_do_not_change_with_the_filters() -> None:
    """The counts take every row; bounds only fade the bins outside them."""
    frame = pl.DataFrame({"key": list("abcde"), "n": [1, 2, 3, 4, 5]})
    config = _config(ColumnConfig("key"), ColumnConfig("n", filter="number"))
    state = parse_state({"t.min__n": ["2"], "t.max__n": ["3.5"]}, config)

    histogram = column_histograms(frame, state, config)["n"]

    assert [(b.count, b.in_filter) for b in histogram.bins] == [
        (1, False),
        (1, True),
        (1, True),
        (1, False),
        (1, False),
    ]


def test_bounds_fade_wide_integer_bins_by_overlap() -> None:
    """A bin of several values stays in while any of them is inside the bounds."""
    frame = pl.DataFrame({"key": ["a", "b"], "n": [0, 99]})
    config = _config(ColumnConfig("n", filter="number"))
    state = parse_state({"t.min__n": ["7"], "t.max__n": ["10"]}, config)

    histogram = column_histograms(frame, state, config)["n"]

    assert [(b.lower, b.upper) for b in histogram.bins if b.in_filter] == [
        ("5", "9"),
        ("10", "14"),
    ]


def test_bounds_fade_float_bins_by_their_half_open_edges() -> None:
    """A float bin ending at the lower bound holds none of its values."""
    frame = pl.DataFrame({"key": ["a", "b"], "x": [0.0, 2.0]})
    config = _config(ColumnConfig("x", filter="number"))
    state = parse_state({"t.min__x": ["1"], "t.max__x": ["1.5"]}, config)

    histogram = column_histograms(frame, state, config)["x"]

    inside = [(b.lower, b.upper) for b in histogram.bins if b.in_filter]
    assert inside[0] == ("1", "1.1")
    assert inside[-1] == ("1.5", "1.6")
    assert not column_histograms(
        frame, parse_state({"t.min__x": ["2"]}, config), config
    )["x"].bins[-2].in_filter
    assert column_histograms(frame, parse_state({"t.min__x": ["2"]}, config), config)[
        "x"
    ].bins[-1].in_filter


def test_datetime_bounds_fade_the_bins_outside_them() -> None:
    """Datetime bounds compare with the bins in the column's type."""
    frame = pl.DataFrame({"key": ["a", "b"], "day": [date(2026, 1, 1), date(2026, 1, 5)]})
    config = _config(ColumnConfig("day", filter="datetime"))
    state = parse_state({"t.min__day": ["2026-01-02T12:00"]}, config)

    histogram = column_histograms(frame, state, config)["day"]

    assert [(b.lower, b.in_filter) for b in histogram.bins] == [
        ("2026-01-01", False),
        ("2026-01-02", False),
        ("2026-01-03", True),
        ("2026-01-04", True),
        ("2026-01-05", True),
    ]


def test_category_columns_show_their_top_values() -> None:
    """The five most frequent values (ties by value) and the rows of the others."""
    values = ["a"] * 4 + ["b"] * 3 + ["c"] * 3 + ["d", "e", "f", "g"] + [None]
    frame = pl.DataFrame(
        {"key": [str(i) for i in range(len(values))], "lot": values},
        schema_overrides={"lot": pl.Categorical},
    )
    config = _config(ColumnConfig("lot", filter="choice"))

    histogram = column_histograms(frame, TableState(), config)["lot"]

    assert histogram == TopValues(
        values=(
            TopValue("a", 4, True),
            TopValue("b", 3, True),
            TopValue("c", 3, True),
            TopValue("d", 1, True),
            TopValue("e", 1, True),
        ),
        others=2,
    )


def test_checked_values_fade_the_others() -> None:
    """Values not checked in the column's filter are faded; counts stay."""
    frame = pl.DataFrame(
        {"key": list("abc"), "ok": [True, False, True]},
    )
    config = _config(ColumnConfig("ok", filter="choice"))
    state = parse_state({"t.eq__ok": ["false"]}, config)

    histogram = column_histograms(frame, state, config)["ok"]

    assert histogram == TopValues(
        values=(TopValue("true", 2, False), TopValue("false", 1, True)), others=0
    )


def test_columns_without_a_distribution_have_none() -> None:
    """Text columns, 128-bit integers, and columns of nulls show no histogram."""
    frame = pl.DataFrame(
        {
            "key": ["a", "b"],
            "name": ["x", "y"],
            "wide": [1, 2],
            "empty": [None, None],
            "nan": [float("nan"), None],
        },
        schema_overrides={"wide": pl.Int128, "empty": pl.Int64},
    )
    config = _config(*(ColumnConfig(name) for name in frame.columns))

    assert column_histograms(frame, TableState(), config) == {}


def test_histograms_of_a_lazy_frame_match_a_data_frame(frame: pl.DataFrame) -> None:
    """A LazyFrame gives the same histograms as its DataFrame, in column order."""
    config = TableConfig(
        table_id="t",
        key="key",
        columns=tuple(
            ColumnConfig(name)
            for name in ("key", "group", "count", "score", "when")
        ),
        url="/t",
    )
    eager = column_histograms(
        frame.with_columns(pl.col("group").cast(pl.Categorical)), TableState(), config
    )
    lazy = column_histograms(
        frame.lazy().with_columns(pl.col("group").cast(pl.Categorical)), TableState(), config
    )

    assert eager == lazy
    assert list(eager) == ["group", "count", "score", "when"]
    assert eager["count"].bins[0] == HistogramBin("1", "1", 1, True)


@pytest.mark.parametrize("histograms", [True, False])
def test_apply_state_shows_histograms_only_when_set(
    frame: pl.DataFrame, config: TableConfig, histograms: bool
) -> None:
    """The view holds the histograms of the whole frame only with ``histograms``."""
    config = replace(config, histograms=histograms)
    state = parse_state({"t.min__count": ["3"]}, config)

    view = apply_state(frame, state, config)

    if histograms:
        assert list(view.histograms) == ["count", "score", "when"]
        assert sum(b.count for b in view.histograms["count"].bins) == 3
    else:
        assert view.histograms == {}
