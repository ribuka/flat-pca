"""Aggregation of PCA loadings by a feature key such as wavelength."""

from __future__ import annotations

from typing import Literal, get_args

import polars as pl

LoadingAggregation = Literal["mean", "rms", "abs_mean"]
LOADING_AGGREGATIONS: tuple[LoadingAggregation, ...] = get_args(LoadingAggregation)


def aggregate_loadings(
    components: pl.DataFrame, method: LoadingAggregation, by: str = "wavelength"
) -> pl.DataFrame:
    """Aggregate component coefficients over the features sharing one key value.

    Parameters
    ----------
    components : pl.DataFrame
        Long-form coefficients with ``component``, ``by``, and
        ``coefficient`` columns, such as the result of
        ``reshape_pca_components``. Other columns are ignored.
    method : LoadingAggregation
        ``"mean"`` for the mean coefficient, ``"rms"`` for the root mean
        square, or ``"abs_mean"`` for the mean absolute value. The mean lets
        coefficients of opposite signs cancel out; the others do not.
    by : str, default ``"wavelength"``
        Column whose values group the features, such as ``"wavelength"``
        or ``"StepTime"``.

    Returns
    -------
    pl.DataFrame
        ``component``, ``by``, and ``loading`` (``Float64``), one row per
        component and key value, sorted by both.

    Raises
    ------
    ValueError
        If ``method`` is unknown or a column is missing.
    """
    coefficient = pl.col("coefficient").cast(pl.Float64)
    expressions = {
        "mean": coefficient.mean(),
        "rms": (coefficient**2).mean().sqrt(),
        "abs_mean": coefficient.abs().mean(),
    }
    if method not in expressions:
        raise ValueError(f"method must be one of {LOADING_AGGREGATIONS}: {method!r}")
    missing = [
        column
        for column in ("component", by, "coefficient")
        if column not in components.columns
    ]
    if missing:
        raise ValueError(f"components lack columns {missing}")
    return (
        components.group_by("component", by)
        .agg(expressions[method].alias("loading"))
        .sort("component", by)
    )
