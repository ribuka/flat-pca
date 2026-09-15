from __future__ import annotations

from typing import Literal

import polars as pl

OutlierStrategy = Literal[None, "winsorize", "drop"]


def _compute_outlier_bounds(
    df: pl.LazyFrame,
    columns: list[str],
    iqr_multiplier: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    stats_exprs: list[pl.Expr] = []
    for col in columns:
        stats_exprs.extend(
            [
                pl.col(col).quantile(0.25).alias(f"{col}__q1"),
                pl.col(col).quantile(0.75).alias(f"{col}__q3"),
                pl.col(col).quantile(0.01).alias(f"{col}__p1"),
                pl.col(col).quantile(0.99).alias(f"{col}__p99"),
            ]
        )

    stats_df = df.select(stats_exprs).collect()
    stats = stats_df.row(0, named=True)

    outlier_lower_bounds: dict[str, float] = {}
    outlier_upper_bounds: dict[str, float] = {}
    winsor_lower_bounds: dict[str, float] = {}
    winsor_upper_bounds: dict[str, float] = {}
    for col in columns:
        q1 = float(stats[f"{col}__q1"])
        q3 = float(stats[f"{col}__q3"])
        iqr = q3 - q1
        outlier_lower_bounds[col] = q1 - iqr_multiplier * iqr
        outlier_upper_bounds[col] = q3 + iqr_multiplier * iqr
        winsor_lower_bounds[col] = float(stats[f"{col}__p1"])
        winsor_upper_bounds[col] = float(stats[f"{col}__p99"])

    return (
        outlier_lower_bounds,
        outlier_upper_bounds,
        winsor_lower_bounds,
        winsor_upper_bounds,
    )


def _prepare_outlier_dataframe(
    df: pl.LazyFrame,
    columns: list[str],
    outlier_strategy: OutlierStrategy,
    iqr_multiplier: float,
    outlier_lower_bounds: dict[str, float] | None = None,
    outlier_upper_bounds: dict[str, float] | None = None,
    winsor_lower_bounds: dict[str, float] | None = None,
    winsor_upper_bounds: dict[str, float] | None = None,
) -> tuple[
    pl.LazyFrame,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    if outlier_strategy is None:
        return df, {}, {}, {}, {}

    resolved_outlier_lower_bounds = outlier_lower_bounds
    resolved_outlier_upper_bounds = outlier_upper_bounds
    resolved_winsor_lower_bounds = winsor_lower_bounds
    resolved_winsor_upper_bounds = winsor_upper_bounds
    if (
        resolved_outlier_lower_bounds is None
        or resolved_outlier_upper_bounds is None
        or resolved_winsor_lower_bounds is None
        or resolved_winsor_upper_bounds is None
    ):
        (
            resolved_outlier_lower_bounds,
            resolved_outlier_upper_bounds,
            resolved_winsor_lower_bounds,
            resolved_winsor_upper_bounds,
        ) = _compute_outlier_bounds(df, columns, iqr_multiplier)

    is_outlier = pl.any_horizontal(
        [
            (pl.col(col) < resolved_outlier_lower_bounds[col])
            | (pl.col(col) > resolved_outlier_upper_bounds[col])
            for col in columns
        ]
    )

    if outlier_strategy == "drop":
        return (
            df.filter(~is_outlier),
            resolved_outlier_lower_bounds,
            resolved_outlier_upper_bounds,
            resolved_winsor_lower_bounds,
            resolved_winsor_upper_bounds,
        )

    return (
        df.with_columns(
            [
                pl.when(is_outlier)
                .then(
                    pl.col(col).clip(
                        resolved_winsor_lower_bounds[col],
                        resolved_winsor_upper_bounds[col],
                    )
                )
                .otherwise(pl.col(col))
                .alias(col)
                for col in columns
            ]
        ),
        resolved_outlier_lower_bounds,
        resolved_outlier_upper_bounds,
        resolved_winsor_lower_bounds,
        resolved_winsor_upper_bounds,
    )
