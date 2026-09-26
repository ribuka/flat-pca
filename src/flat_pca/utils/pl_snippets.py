"""Polars helpers shared by the workflow modules and analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs

from .natural_keys import natural_keys

HOW = Literal["left", "right", "outer", "full", "inner"]


def _normalize_join_how(how: HOW) -> Literal["left", "right", "full", "inner"]:
    """Translate a join strategy to the name current Polars accepts.

    Parameters
    ----------
    how : HOW
        Requested join strategy, possibly the deprecated ``"outer"``.

    Returns
    -------
    Literal["left", "right", "full", "inner"]
        Equivalent strategy name, with ``"outer"`` mapped to ``"full"``.
    """
    return "full" if how == "outer" else how


def _join_polars_with_indicator(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    on: list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: HOW = "left",
    indicator: str | bool = "_merge",
) -> pl.DataFrame | pl.LazyFrame:
    """Join two Polars frames and optionally add a merge-origin indicator.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to join. Both must be of the same kind.
    on : list[str] | None, default None
        Shared key columns. Takes precedence over ``left_on``/``right_on``.
    left_on, right_on : str | list[str] | None, default None
        Per-side key columns, used when ``on`` is empty or ``None``.
    how : HOW, default "left"
        Join strategy; ``"outer"`` is accepted as a synonym of ``"full"``.
    indicator : str | bool, default "_merge"
        Name of the merge-origin column holding ``"both"``, ``"left_only"``,
        or ``"right_only"``. Pass ``False`` to omit the column.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Joined frame, of the same kind as the inputs.

    Raises
    ------
    ValueError
        If neither ``on`` nor both of ``left_on`` and ``right_on`` is given.
    """
    left = left.with_columns(pl.lit(True).alias("in_left"))
    right = right.with_columns(pl.lit(True).alias("in_right"))
    how = _normalize_join_how(how)

    if on:
        out = left.join(right, on=on, how=how)  # ty:ignore[invalid-argument-type]
    elif left_on and right_on:
        out = left.join(right, left_on=left_on, right_on=right_on, how=how)  # ty:ignore[invalid-argument-type]
    else:
        raise ValueError("Invalid 'on' parameter")

    out = out.with_columns([
        pl.col("in_left").fill_null(False),
        pl.col("in_right").fill_null(False),
    ])

    if indicator is not False:
        out = out.with_columns([
            pl.when(pl.col("in_left") & pl.col("in_right")).then(pl.lit("both"))
            .when(pl.col("in_left")).then(pl.lit("left_only"))
            .otherwise(pl.lit("right_only"))
            .alias(indicator)
        ])

    out = out.drop("in_left", "in_right")

    return out


def my_merge(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    addons: list[str] | None = None,
    on: list[str] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: Literal["left", "right", "outer", "full", "inner"] = "left",
    indicator: str | bool = "_merge",
    show_result: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Merge two same-kind Polars frames with optional column selection.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to merge. Both must be of the same kind.
    addons : list[str] | None, default None
        Columns to take from ``right``. If ``None``, every column is taken.
    on : list[str] | None, default None
        Shared key columns, defaulting to ``["ID"]``.
    left_on, right_on : str | None, default None
        Per-side key columns, used when ``on`` is empty.
    how : Literal["left", "right", "outer", "full", "inner"], default "left"
        Join strategy; ``"outer"`` is accepted as a synonym of ``"full"``.
    indicator : str | bool, default "_merge"
        Name of the merge-origin column, or ``False`` to omit it.
    show_result : bool, default True
        Whether to print the shape change and the merge-origin counts, which
        is intended for interactive notebook use.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Merged frame, of the same kind as the inputs.

    Raises
    ------
    TypeError
        If ``left`` and ``right`` are not of the same kind.
    ValueError
        If ``on`` is empty and ``left_on``/``right_on`` are not both given.
    """
    if on is None:
        on = ["ID"]

    if type(left) is not type(right):
        raise TypeError(f"df: {type(left)} and wfl: {type(right)} must be the same type")


    if (
        isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)
    ) or (
        isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)
    ):
        if show_result:
            _shape = get_shape_from_polars(left)

        if addons is None:
            addons = get_columns_from_polars(right)
            # addons.remove(on) if on is not None else addons.remove(right_on)  # ty:ignore[invalid-argument-type]

        if on:
            left = _join_polars_with_indicator(
                left, right.select(addons), on=on, how=how, indicator=indicator,
            )

        else:
            if not left_on or not right_on:
                raise ValueError("`left_on` and `right_on` must be specified if `on` is None")

            if right_on not in addons:
                addons.append(right_on)

            left = _join_polars_with_indicator(
                left, right.select(addons), on=[], left_on=left_on, right_on=right_on, how=how, indicator=indicator,
            )

        if show_result:
            print(f"{_shape} -> {get_shape_from_polars(left)}")
            if indicator is not False:
                print(get_value_counts_from_polars(
                    left, alias=indicator,
                ))

    return left


def get_value_counts_from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str | bool,
    sort: bool = True,
) -> pl.DataFrame:
    """Count occurrences of each distinct value in one column.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column to summarize.
    alias : str | bool
        Name of the column to count.
    sort : bool, default True
        Whether to sort the result by descending count.

    Returns
    -------
    pl.DataFrame
        Distinct values with their ``count``.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.LazyFrame):
        # vc = df.collect()[alias].value_counts()
        vc = df.select(alias).collect()[alias].value_counts()  # ty:ignore[not-subscriptable, unresolved-attribute]
    elif isinstance(df, pl.DataFrame):
        vc = df[alias].value_counts()  # ty:ignore[unresolved-attribute]
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")
    if sort:
        vc = vc.sort("count", descending=True)
    return vc


def get_shape_from_polars(
    df: pl.DataFrame | pl.LazyFrame
) -> tuple[int, int]:
    """Return a frame's row and column counts.

    For a ``LazyFrame`` the row count is obtained from a ``pl.len()``
    aggregation and the column count from the resolved schema, so the frame
    itself is never materialized.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to measure.

    Returns
    -------
    tuple[int, int]
        Number of rows and number of columns.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.DataFrame):
        return df.shape
    elif isinstance(df, pl.LazyFrame):
        n_rows = df.select(pl.len().alias("n_rows")).collect()["n_rows"][0]  # ty:ignore[not-subscriptable]
        n_cols = len(df.collect_schema())
        return (n_rows, n_cols)
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


def drop_list_type_columns_from_polars(
    df: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame | pl.LazyFrame:
    """Drop every List-typed column, which most writers cannot serialize.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to inspect.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``df`` without its List-typed columns. The dropped names are printed
        so notebook callers can see what was removed.
    """
    drop_list = [
        column
        for column, dtype in get_schema_from_polars(df).items()
        if dtype == pl.datatypes.List
    ]
    print(f"{drop_list = }")

    return df.drop(drop_list)


def validate_missing_ratio_threshold(threshold: float) -> None:
    """Validate a missing-value ratio threshold.

    Parameters
    ----------
    threshold : float
        Inclusive missing-value ratio threshold to validate.

    Raises
    ------
    ValueError
        If ``threshold`` is not between 0.0 and 1.0.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")


def _count_missing_values(
    frame: pl.DataFrame | pl.LazyFrame,
    include_nan_missing: bool,
) -> tuple[int, dict[str, int]]:
    """Count the rows and each column's missing values in bulk.

    Nulls are counted by ``null_count`` and, when requested, NaNs by one
    selector over the floating-point columns, instead of one aggregation
    expression per column, which dominates the cost at six-figure widths.
    ``is_nan`` yields null for null entries and ``sum`` skips them, so no
    entry is counted twice. A ``pl.DataFrame`` is aggregated eagerly,
    because routing it through a lazy query costs far more than the
    aggregation itself on wide frames.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated. A ``pl.LazyFrame`` collects only
        the statistics queries.
    include_nan_missing : bool
        Whether NaN values count as missing for floating-point columns.

    Returns
    -------
    tuple[int, dict[str, int]]
        Row count and missing-value count per column, in column order.
    """
    nan_count_expr = cs.float().is_nan().sum()
    if isinstance(frame, pl.LazyFrame):
        queries = [frame.select(pl.len()), frame.null_count()]
        if include_nan_missing:
            queries.append(frame.select(nan_count_expr))
        stats = pl.collect_all(queries)
    else:
        stats = [frame.select(pl.len()), frame.null_count()]
        if include_nan_missing:
            stats.append(frame.select(nan_count_expr))
    row_count = stats[0].item()
    missing_counts, *nan_counts = [
        stat.row(0, named=True) if stat.width else {} for stat in stats[1:]
    ]
    for col_name, nan_count in (nan_counts[0] if nan_counts else {}).items():
        missing_counts[col_name] += nan_count
    return row_count, missing_counts


def drop_all_null_columns_from_polars(
    frame: pl.DataFrame | pl.LazyFrame,
    include_nan_missing: bool = True,
    threshold: float = 0.99,
) -> pl.DataFrame | pl.LazyFrame:
    """Drop columns whose missing-value ratio exceeds a threshold.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated for missing values.
    include_nan_missing : bool, default True
        Whether NaN values count as missing for floating-point columns.
    threshold : float, default 0.99
        Inclusive maximum missing-value ratio for retained columns.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``frame`` restricted to columns at or below ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is not between 0.0 and 1.0.
    """
    validate_missing_ratio_threshold(threshold)

    row_count, missing_counts = _count_missing_values(frame, include_nan_missing)
    if row_count == 0:
        return frame

    keep_cols = [
        col_name
        for col_name, missing_count in missing_counts.items()
        if missing_count / row_count <= threshold
    ]

    return frame.select(keep_cols)


def safe_write_csv(
    df: pl.DataFrame, file_path: str | Path
) -> None:
    """Write a frame to CSV after dropping columns CSV cannot represent.

    Parameters
    ----------
    df : pl.DataFrame
        Frame to write. List-typed columns are dropped first.
    file_path : str | Path
        Destination CSV path.
    """
    df = drop_list_type_columns_from_polars(df)  # ty:ignore[invalid-assignment]
    df.write_csv(file_path)
    print(f"Written csv to {file_path}")


def get_columns_from_polars(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return column names without materializing a LazyFrame.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose schema supplies the column names.

    Returns
    -------
    list[str]
        Frame column names in their existing order.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.DataFrame):
        return df.columns
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


def get_schema_from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    args: list[str] | None = None,
) -> dict[str, pl.DataType]:
    """Return the schema of a Polars frame, optionally limited to columns.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose schema is resolved. A ``LazyFrame`` is not materialized.
    args : list[str] | None, default None
        Columns to restrict the schema to. If ``None`` or empty, the full
        schema is returned.

    Returns
    -------
    dict[str, pl.DataType]
        Column names mapped to their dtypes.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if args:
        df = df.select(args)
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema()
    if isinstance(df, pl.DataFrame):
        return df.schema
    raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


def get_numeric_args(
    df: pl.DataFrame,
    args: list[str] | None = None,
) -> list[str]:
    """Return numeric column names from a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Frame whose schema is inspected.
    args : list[str] | None, default None
        Candidate column names. If ``None`` or empty, every column is
        considered.

    Returns
    -------
    list[str]
        Names of the candidate columns with a numeric dtype.
    """
    if not args:
        args = get_columns_from_polars(df)
    numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    mapping = get_schema_from_polars(df, args)
    numerical_args = [key for key, dtype in mapping.items() if dtype in numeric_types]
    # if not numerical_args:
    #     raise ValueError("No numerical columns found.")
    return numerical_args


def get_unique_values_from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str,
    sort: bool = True,
) -> list[str]:
    """Return one column's distinct non-null values.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column.
    alias : str
        Name of the column to read.
    sort : bool, default True
        Whether to sort the values with ``natural_keys``, so that embedded
        numbers order numerically rather than lexicographically.

    Returns
    -------
    list[str]
        Distinct values of the column.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.LazyFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).collect().to_series().to_list()
    elif isinstance(df, pl.DataFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).to_series().to_list()
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")

    if sort:
        return sorted(uniques, key=natural_keys)
    return uniques
