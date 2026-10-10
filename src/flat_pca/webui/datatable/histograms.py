"""The distributions of a data table's columns shown under the column names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import polars as pl

from .bounds import bound_condition
from .config import TableConfig
from .formatting import format_value
from .state import Bound, TableState

# Most bins of a histogram.
MAX_BINS = 20
# Most values with a bar of a category column.
TOP_VALUES = 5
# How each temporal type is written on a histogram.
TEMPORAL_FORMATS: dict[type[pl.DataType], str] = {
    pl.Date: "%Y-%m-%d",
    pl.Datetime: "%Y-%m-%d %H:%M:%S",
}

type BinKind = Literal["integer", "float"]


@dataclass(frozen=True)
class HistogramBin:
    """One bar of a histogram.

    Attributes
    ----------
    lower, upper : str
        First and last value of the bin as text. An integer or temporal bin
        holds the values from ``lower`` to ``upper``, both included; a float
        bin holds those from ``lower`` up to ``upper``, which only the last
        bin includes.
    count : int
        Rows whose value is in the bin.
    in_filter : bool
        Whether the bin overlaps the column's lower and upper bounds in use
        (always ``True`` without bounds); the others are shown faded.
    """

    lower: str
    upper: str
    count: int
    in_filter: bool


@dataclass(frozen=True)
class Histogram:
    """Distribution of a number or temporal column over every row.

    Attributes
    ----------
    bins : tuple[HistogramBin, ...]
        Bins from the smallest value to the largest, of equal width.
    minimum, maximum : str
        Smallest and largest value as text, written under the bars.
    """

    bins: tuple[HistogramBin, ...]
    minimum: str
    maximum: str


@dataclass(frozen=True)
class TopValue:
    """One bar of the most frequent values of a category column.

    Attributes
    ----------
    value : str
        Value as text.
    count : int
        Rows with the value.
    in_filter : bool
        Whether the value is among the values checked in the column's filter
        (always ``True`` when none is checked); the others are shown faded.
    """

    value: str
    count: int
    in_filter: bool


@dataclass(frozen=True)
class TopValues:
    """Most frequent values of a category column over every row.

    Attributes
    ----------
    values : tuple[TopValue, ...]
        Up to ``TOP_VALUES`` values, the most frequent first (ties by value).
    others : int
        Rows with a non-null value that is not listed.
    """

    values: tuple[TopValue, ...]
    others: int


type ColumnHistogram = Histogram | TopValues


def bin_kind(dtype: pl.DataType) -> BinKind | None:
    """Return how a column type is cut into histogram bins.

    Parameters
    ----------
    dtype : pl.DataType
        Polars type of the column.

    Returns
    -------
    BinKind | None
        ``"integer"`` for integers up to 64 bits, dates, and datetimes
        (binned by their integer physical value), ``"float"`` for floats,
        and ``None`` for types without a histogram.
    """
    if dtype in (pl.Int128, pl.UInt128):
        return None
    if dtype.is_integer() or isinstance(dtype, (pl.Date, pl.Datetime)):
        return "integer"
    if dtype.is_float():
        return "float"
    return None


def has_top_values(dtype: pl.DataType) -> bool:
    """Return whether a column type shows its most frequent values.

    Parameters
    ----------
    dtype : pl.DataType
        Polars type of the column.

    Returns
    -------
    bool
        Whether the type is ``pl.Categorical``, ``pl.Enum``, or
        ``pl.Boolean``.
    """
    return isinstance(dtype, (pl.Categorical, pl.Enum, pl.Boolean))


def integer_bins(minimum: int, maximum: int) -> list[tuple[int, int]]:
    """Cut a range of integers into bins of equal width.

    Parameters
    ----------
    minimum, maximum : int
        Smallest and largest value, ``minimum <= maximum``.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive ``(first, last)`` of each bin: one bin per value when the
        range has at most ``MAX_BINS`` values, otherwise bins of
        ``ceil(values / MAX_BINS)`` values (at most ``MAX_BINS`` of them, the
        last one ending at ``maximum``).
    """
    # Integer division keeps every digit of a 64-bit range.
    width = -(-(maximum - minimum + 1) // MAX_BINS)
    return [
        (first, min(first + width - 1, maximum))
        for first in range(minimum, maximum + 1, width)
    ]


def float_bins(minimum: float, maximum: float) -> list[tuple[float, float]]:
    """Cut a range of floats into bins of equal width.

    Parameters
    ----------
    minimum, maximum : float
        Smallest and largest finite value, ``minimum <= maximum``.

    Returns
    -------
    list[tuple[float, float]]
        ``(lower, upper)`` edges of ``MAX_BINS`` bins from ``minimum`` to
        ``maximum``, or of one bin when they are equal. The edges are
        interpolated without computing ``maximum - minimum``, so that they
        stay finite over the whole float range, and never decrease.
    """
    if minimum == maximum:
        return [(minimum, maximum)]
    edges = [minimum]
    for index in range(1, MAX_BINS):
        share = index / MAX_BINS
        edge = minimum * (1 - share) + maximum * share
        edges.append(min(max(edge, edges[-1]), maximum))
    edges.append(maximum)
    return list(pairwise(edges))


def _binned_values(name: str, kind: BinKind) -> pl.Expr:
    """Return the values a column is binned by (null for none).

    Integer kinds give their physical value as ``Int128``, which holds the
    distance between any two 64-bit values; floats give their finite values.
    """
    if kind == "integer":
        return pl.col(name).to_physical().cast(pl.Int128)
    column = pl.col(name).cast(pl.Float64)
    return pl.when(column.is_finite()).then(column)


def _bin_index(
    name: str, kind: BinKind, bins: list[tuple[int, int]] | list[tuple[float, float]]
) -> pl.Expr:
    """Return the 0-based bin of each value of a column (null for no bin).

    A float value goes to the last bin whose lower edge is not above it, by
    the same edges that the bins show and that the bounds are compared with.
    """
    values = _binned_values(name, kind)
    if kind == "integer":
        width = bins[0][1] - bins[0][0] + 1
        return (values - pl.lit(bins[0][0], dtype=pl.Int128)) // width
    inner = pl.Series([lower for lower, _ in bins[1:]], dtype=pl.Float64)
    index = pl.lit(inner).search_sorted(values, side="right").cast(pl.Int128)
    return pl.when(values.is_not_null()).then(index)


def _physical(dtype: pl.DataType) -> pl.DataType:
    """Return the integer type a binned column is stored as."""
    if isinstance(dtype, pl.Date):
        return pl.Int32()
    if isinstance(dtype, pl.Datetime):
        return pl.Int64()
    return dtype


def _value_texts(values: list[int] | list[float], dtype: pl.DataType) -> list[str]:
    """Return bin edges, given as physical values of a column type, as text."""
    if not dtype.is_temporal():
        return [format_value(value) for value in values]
    series = pl.Series(values, dtype=pl.Int64).cast(_physical(dtype)).cast(dtype)
    return series.dt.to_string(TEMPORAL_FORMATS[dtype.base_type()]).to_list()


def _in_bounds(
    bins: list[tuple[int, int]] | list[tuple[float, float]],
    kind: BinKind,
    dtype: pl.DataType,
    bounds: tuple[Bound, Bound],
) -> list[bool]:
    """Return whether each bin overlaps inclusive bounds (``None`` for open).

    A float bin ``[lower, upper)`` overlaps when ``upper > lower bound`` (the
    last bin, which includes ``upper``, when ``upper >= lower bound``) and
    ``lower <= upper bound``. An integer or temporal bin is compared in the
    column's own type, as the filter compares the values.
    """
    low, high = bounds
    if low is None and high is None:
        return [True] * len(bins)
    if kind == "float":
        last = len(bins) - 1
        return [
            (low is None or upper > low or (index == last and upper >= low))  # type: ignore[operator]
            and (high is None or lower <= high)  # type: ignore[operator]
            for index, (lower, upper) in enumerate(bins)
        ]
    edges = pl.DataFrame(
        {"first": [first for first, _ in bins], "last": [last for _, last in bins]},
        schema={"first": pl.Int128, "last": pl.Int128},
    )
    edges = edges.cast(_physical(dtype)).cast(dtype)
    conditions = []
    if low is not None:
        conditions.append(bound_condition("last", low, dtype, lower=True))
    if high is not None:
        conditions.append(bound_condition("first", high, dtype, lower=False))
    return edges.select(pl.all_horizontal(conditions).fill_null(False)).to_series().to_list()


def column_histograms(
    frame: pl.DataFrame | pl.LazyFrame,
    state: TableState,
    config: TableConfig,
    schema: Mapping[str, pl.DataType] | None = None,
) -> dict[str, ColumnHistogram]:
    """Summarize the distribution of each shown column over every row.

    The distributions do not depend on the filters, so they stay the same
    while the filters change; the filters only fade the bins outside the
    bounds and the values not checked.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table.
    state : TableState
        State whose bounds (``ranges``) and checked values (``equals``) fade
        the bins and values outside them.
    config : TableConfig
        Table settings.
    schema : Mapping[str, pl.DataType] | None, default None
        Column types of the frame; collected from it if ``None``.

    Returns
    -------
    dict[str, ColumnHistogram]
        By column name, a ``Histogram`` of each number, date, and datetime
        column (``bin_kind``; NaN and infinite floats are not counted) and
        the ``TopValues`` of each categorical, enum, and boolean column
        (``has_top_values``). A column without a non-null value has none.
    """
    lazy = frame.lazy()
    if schema is None:
        schema = lazy.collect_schema()
    binned = {
        column.name: kind
        for column in config.columns
        if (kind := bin_kind(schema[column.name])) is not None
    }
    topped = [column.name for column in config.columns if has_top_values(schema[column.name])]
    if not binned and not topped:
        return {}

    # Positional names keep the results apart from any column name.
    ranges = lazy.select(
        *(
            _binned_values(name, kind).min().alias(f"min{index}")
            for index, (name, kind) in enumerate(binned.items())
        ),
        *(
            _binned_values(name, kind).max().alias(f"max{index}")
            for index, (name, kind) in enumerate(binned.items())
        ),
        *(
            pl.col(name)
            .cast(pl.String)
            .alias("value")
            .drop_nulls()
            .value_counts(name="n")
            .implode()
            .alias(f"values{index}")
            for index, name in enumerate(topped)
        ),
    ).collect().row(0, named=True)

    cuts: dict[str, list[tuple[int, int]] | list[tuple[float, float]]] = {}
    for index, (name, kind) in enumerate(binned.items()):
        minimum, maximum = ranges[f"min{index}"], ranges[f"max{index}"]
        if minimum is not None:
            cuts[name] = (
                integer_bins(minimum, maximum) if kind == "integer" else float_bins(minimum, maximum)
            )
    counted = (
        lazy.select(
            _bin_index(name, binned[name], bins)
            .alias("bin")
            .drop_nulls()
            .value_counts(name="n")
            .implode()
            .alias(f"bins{index}")
            for index, (name, bins) in enumerate(cuts.items())
        )
        .collect()
        .row(0, named=True)
        if cuts
        else {}
    )

    histograms: dict[str, ColumnHistogram] = {}
    for index, (name, bins) in enumerate(cuts.items()):
        dtype, kind = schema[name], binned[name]
        counts = {pair["bin"]: pair["n"] for pair in counted[f"bins{index}"]}
        lowers = _value_texts([lower for lower, _ in bins], dtype)
        uppers = _value_texts([upper for _, upper in bins], dtype)
        in_filter = _in_bounds(bins, kind, dtype, state.ranges.get(name, (None, None)))
        histograms[name] = Histogram(
            bins=tuple(
                HistogramBin(lowers[bin], uppers[bin], counts.get(bin, 0), in_filter[bin])
                for bin in range(len(bins))
            ),
            minimum=lowers[0],
            maximum=uppers[-1],
        )
    for index, name in enumerate(topped):
        pairs = sorted(
            ((pair["value"], pair["n"]) for pair in ranges[f"values{index}"]),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if not pairs:
            continue
        chosen = state.equals.get(name)
        histograms[name] = TopValues(
            values=tuple(
                TopValue(value, count, chosen is None or value in chosen)
                for value, count in pairs[:TOP_VALUES]
            ),
            others=sum(count for _, count in pairs[TOP_VALUES:]),
        )
    order = [column.name for column in config.columns]
    return {name: histograms[name] for name in order if name in histograms}
