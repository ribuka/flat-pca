"""IQR-based outlier detection and handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

OutlierStrategy = Literal["winsorize", "drop"] | None


@dataclass(frozen=True)
class OutlierBounds:
    """Store per-column thresholds used to detect and clip outliers.

    Attributes
    ----------
    outlier_lower : dict[str, float]
        Per-column lower outlier thresholds, ``Q1 - multiplier * IQR``.
    outlier_upper : dict[str, float]
        Per-column upper outlier thresholds, ``Q3 + multiplier * IQR``.
    winsor_lower : dict[str, float]
        Per-column lower clipping bounds, the 1st percentile.
    winsor_upper : dict[str, float]
        Per-column upper clipping bounds, the 99th percentile.
    """

    outlier_lower: dict[str, float]
    outlier_upper: dict[str, float]
    winsor_lower: dict[str, float]
    winsor_upper: dict[str, float]

    @classmethod
    def empty(cls) -> OutlierBounds:
        """Return bounds holding no thresholds, used when handling is disabled.

        Returns
        -------
        OutlierBounds
            Bounds whose four mappings are all empty.
        """
        return cls(
            outlier_lower={},
            outlier_upper={},
            winsor_lower={},
            winsor_upper={},
        )


def compute_outlier_bounds(
    df: pl.LazyFrame,
    columns: list[str],
    iqr_multiplier: float,
) -> OutlierBounds:
    """Compute IQR outlier thresholds and percentile clipping bounds.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to compute thresholds for.
    iqr_multiplier : float
        Positive multiplier applied to the interquartile range.

    Returns
    -------
    OutlierBounds
        Per-column outlier thresholds and clipping bounds.
    """
    stats_exprs: list[pl.Expr] = []
    for column in columns:
        stats_exprs.extend(
            [
                pl.col(column).quantile(0.25).alias(f"{column}__q1"),
                pl.col(column).quantile(0.75).alias(f"{column}__q3"),
                pl.col(column).quantile(0.01).alias(f"{column}__p1"),
                pl.col(column).quantile(0.99).alias(f"{column}__p99"),
            ]
        )

    stats = df.select(stats_exprs).collect().row(0, named=True)

    outlier_lower: dict[str, float] = {}
    outlier_upper: dict[str, float] = {}
    winsor_lower: dict[str, float] = {}
    winsor_upper: dict[str, float] = {}
    for column in columns:
        q1 = float(stats[f"{column}__q1"])
        q3 = float(stats[f"{column}__q3"])
        iqr = q3 - q1
        outlier_lower[column] = q1 - iqr_multiplier * iqr
        outlier_upper[column] = q3 + iqr_multiplier * iqr
        winsor_lower[column] = float(stats[f"{column}__p1"])
        winsor_upper[column] = float(stats[f"{column}__p99"])

    return OutlierBounds(
        outlier_lower=outlier_lower,
        outlier_upper=outlier_upper,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
    )


def prepare_outlier_frame(
    df: pl.LazyFrame,
    columns: list[str],
    outlier_strategy: OutlierStrategy,
    iqr_multiplier: float,
    bounds: OutlierBounds | None = None,
) -> tuple[pl.LazyFrame, OutlierBounds]:
    """Apply the configured outlier handling and return the bounds used.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric columns to inspect for outliers.
    outlier_strategy : OutlierStrategy
        ``None`` to leave the data untouched, ``"drop"`` to remove rows with
        any outlier value, or ``"winsorize"`` to clip those rows' values to
        the percentile bounds.
    iqr_multiplier : float
        Positive multiplier used when thresholds must be computed.
    bounds : OutlierBounds | None, default None
        Previously fitted thresholds to reuse. If ``None``, thresholds are
        computed from ``df`` itself.

    Returns
    -------
    tuple[pl.LazyFrame, OutlierBounds]
        Transformed data and the thresholds that were applied. Both are
        empty-valued when ``outlier_strategy`` is ``None``.
    """
    if outlier_strategy is None:
        return df, OutlierBounds.empty()

    resolved_bounds = (
        compute_outlier_bounds(df, columns, iqr_multiplier)
        if bounds is None
        else bounds
    )

    is_outlier = pl.any_horizontal(
        [
            (pl.col(column) < resolved_bounds.outlier_lower[column])
            | (pl.col(column) > resolved_bounds.outlier_upper[column])
            for column in columns
        ]
    )

    if outlier_strategy == "drop":
        return df.filter(~is_outlier), resolved_bounds

    return (
        df.with_columns(
            [
                pl.when(is_outlier)
                .then(
                    pl.col(column).clip(
                        resolved_bounds.winsor_lower[column],
                        resolved_bounds.winsor_upper[column],
                    )
                )
                .otherwise(pl.col(column))
                .alias(column)
                for column in columns
            ]
        ),
        resolved_bounds,
    )
