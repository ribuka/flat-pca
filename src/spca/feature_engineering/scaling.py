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


def _collect_to_df(lf: pl.LazyFrame) -> pl.DataFrame:
    obj = lf.collect()
    return obj if isinstance(obj, pl.DataFrame) else obj.collect()


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
            centers={col: 0.0 for col in columns},
            scales={col: 1.0 for col in columns},
        )
    if strategy == "z-score":
        stats_df = _collect_to_df(
            df.select(
                [pl.col(col).mean().alias(f"{col}__center") for col in columns]
                + [pl.col(col).std().alias(f"{col}__scale") for col in columns]
            )
        )
    elif strategy == "minmax":
        stats_df = _collect_to_df(
            df.select(
                [pl.col(col).min().alias(f"{col}__center") for col in columns]
                + [
                    (pl.col(col).max() - pl.col(col).min()).alias(f"{col}__scale")
                    for col in columns
                ]
            )
        )
    else:
        stats_df = _collect_to_df(
            df.select(
                [pl.col(col).median().alias(f"{col}__center") for col in columns]
                + [
                    (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(
                        f"{col}__scale"
                    )
                    for col in columns
                ]
            )
        )

    row = stats_df.row(0, named=True)
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for col in columns:
        center_raw = row[f"{col}__center"]
        scale_raw = row[f"{col}__scale"]
        centers[col] = float(center_raw) if center_raw is not None else 0.0
        scale_value = float(scale_raw) if scale_raw is not None else 1.0
        scales[col] = scale_value if scale_value != 0 else 1.0

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

    exprs = [
        (
            (pl.col(col).cast(pl.Float64) - scaling_model.centers[col])
            / scaling_model.scales[col]
        ).alias(col)
        for col in columns
    ]
    return df.with_columns(exprs)


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
