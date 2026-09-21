from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from .natural_keys import natural_keys

HOW = Literal["left", "right", "outer", "full", "inner"]


def _normalize_join_how(how: HOW) -> Literal["left", "right", "full", "inner"]:
    # Polars deprecated how="outer" in favor of how="full".
    return "full" if how == "outer" else how


def _join_polars_with_indicator(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    on: list[str] | None = None,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
    how: HOW = "left",
    indicator: str | bool = "_merge",
) -> pl.DataFrame | pl.LazyFrame:
    """Join two Polars frames and optionally add a merge-origin indicator."""
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

    if indicator != False:
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
    """Merge two same-kind Polars frames with optional column selection."""
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
            if indicator != False:
                print(get_value_counts_from_polars(
                    left, alias=indicator,
                ))

    return left


def get_value_counts_from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str | bool,
    sort: bool = True,
) -> pl.DataFrame:
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
) -> tuple:
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
    drop_list = [
        c for c, dtype in zip(df.columns, df.dtypes)
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

    schema = (
        frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    )
    missing_count_exprs = []
    for col_name, dtype in schema.items():
        is_missing = pl.col(col_name).is_null()
        if include_nan_missing and dtype in (pl.Float32, pl.Float64):
            is_missing = is_missing | pl.col(col_name).is_nan()
        missing_count_exprs.append(is_missing.sum().alias(f"{col_name}_missing"))

    stats = frame.select(
        *missing_count_exprs,
        pl.len().alias("_n")
    )
    if isinstance(stats, pl.LazyFrame):
        stats = stats.collect()
    n = stats[0, "_n"]  # ty:ignore[not-subscriptable]
    if n == 0:
        return frame

    keep_cols = [
        col_name
        for col_name in schema.names()
        if (stats[0, f"{col_name}_missing"] / n) <= threshold  # ty:ignore[unresolved-attribute, not-subscriptable]
    ]

    return frame.select(keep_cols)


def safe_write_csv(
    df: pl.DataFrame, file_path: str | Path
) -> None:
    df = drop_list_type_columns_from_polars(df)  # ty:ignore[invalid-assignment]
    df.write_csv(file_path)
    print(f"Written csv to {file_path}")


def get_columns_from_polars(df) -> list:
    if isinstance(df, pl.DataFrame):
        return df.columns
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


def get_schema_from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    args: list[str] | None = None,
) -> dict[str, pl.DataType]:
    """Return the schema of a Polars frame, optionally limited to columns."""
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
    """Return numeric column names from a DataFrame."""
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
    if isinstance(df, pl.LazyFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).collect().to_series().to_list()
    elif isinstance(df, pl.DataFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).to_series().to_list()
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")

    if sort:
        return sorted(uniques, key=natural_keys)
    return uniques
