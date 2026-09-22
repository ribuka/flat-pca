"""Column scaling strategies fitted and applied over Polars frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

ScalingStrategy = Literal["none", "z-score", "minmax", "robust"]


@dataclass(frozen=True)
class ScalingModel:
    """Store fitted column scaling parameters.

    Attributes
    ----------
    strategy : ScalingStrategy
        Scaling transformation represented by the model.
    centers : dict[str, float]
        Per-column values subtracted before scaling.
    scales : dict[str, float]
        Per-column nonzero divisors used for scaling.
    """

    strategy: ScalingStrategy
    centers: dict[str, float]
    scales: dict[str, float]


def _center_and_scale_exprs(
    columns: list[str],
    strategy: ScalingStrategy,
) -> list[pl.Expr]:
    """Build the per-column center and scale aggregations of one strategy.

    Parameters
    ----------
    columns : list[str]
        Numeric columns to aggregate.
    strategy : ScalingStrategy
        Scaling strategy whose statistics are needed. ``"none"`` has no
        statistics and must be handled by the caller.

    Returns
    -------
    list[pl.Expr]
        One ``{column}__center`` and one ``{column}__scale`` expression per
        column.
    """
    if strategy == "z-score":
        centers = [pl.col(column).mean() for column in columns]
        scales = [pl.col(column).std() for column in columns]
    elif strategy == "minmax":
        centers = [pl.col(column).min() for column in columns]
        scales = [pl.col(column).max() - pl.col(column).min() for column in columns]
    else:
        centers = [pl.col(column).median() for column in columns]
        scales = [
            pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25)
            for column in columns
        ]

    return [
        center.alias(f"{column}__center")
        for column, center in zip(columns, centers, strict=True)
    ] + [
        scale.alias(f"{column}__scale")
        for column, scale in zip(columns, scales, strict=True)
    ]


def fit_scaler(
    df: pl.LazyFrame,
    columns: list[str],
    strategy: ScalingStrategy,
) -> ScalingModel:
    """Fit scaling parameters for selected columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns used to fit the scaler.
    strategy : ScalingStrategy
        Scaling strategy to fit.

    Returns
    -------
    ScalingModel
        Fitted centers, scales, and strategy.
    """
    if not columns:
        return ScalingModel(strategy=strategy, centers={}, scales={})

    if strategy == "none":
        return ScalingModel(
            strategy=strategy,
            centers={column: 0.0 for column in columns},
            scales={column: 1.0 for column in columns},
        )

    row = df.select(_center_and_scale_exprs(columns, strategy)).collect().row(
        0, named=True
    )
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        center_raw = row[f"{column}__center"]
        scale_raw = row[f"{column}__scale"]
        centers[column] = float(center_raw) if center_raw is not None else 0.0
        scale_value = float(scale_raw) if scale_raw is not None else 1.0
        scales[column] = scale_value if scale_value != 0 else 1.0

    return ScalingModel(strategy=strategy, centers=centers, scales=scales)


def apply_scaler(
    df: pl.LazyFrame,
    columns: list[str],
    scaling_model: ScalingModel,
) -> pl.LazyFrame:
    """Apply fitted scaling parameters to selected columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to transform.
    scaling_model : ScalingModel
        Previously fitted scaling parameters.

    Returns
    -------
    pl.LazyFrame
        Input data with the selected columns transformed.
    """
    if not columns or scaling_model.strategy == "none":
        return df

    return df.with_columns(
        [
            (
                (pl.col(column).cast(pl.Float64) - scaling_model.centers[column])
                / scaling_model.scales[column]
            ).alias(column)
            for column in columns
        ]
    )


def zscore_standardize(
    df: pl.LazyFrame,
    columns: list[str],
) -> pl.LazyFrame:
    """Apply z-score standardization to selected columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to standardize.

    Returns
    -------
    pl.LazyFrame
        Input data with standardized selected columns.
    """
    return apply_scaler(df, columns, fit_scaler(df, columns, "z-score"))


def minmax_scale(
    df: pl.LazyFrame,
    columns: list[str],
) -> pl.LazyFrame:
    """Apply min-max scaling to selected columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to scale.

    Returns
    -------
    pl.LazyFrame
        Input data with min-max-scaled selected columns.
    """
    return apply_scaler(df, columns, fit_scaler(df, columns, "minmax"))


def robust_scale(
    df: pl.LazyFrame,
    columns: list[str],
) -> pl.LazyFrame:
    """Apply robust scaling using median and IQR to selected columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to scale.

    Returns
    -------
    pl.LazyFrame
        Input data with robust-scaled selected columns.
    """
    return apply_scaler(df, columns, fit_scaler(df, columns, "robust"))
