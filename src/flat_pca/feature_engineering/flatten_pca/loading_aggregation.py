"""Aggregation of PCA loadings by wavelength."""

from __future__ import annotations

from typing import Literal, get_args

import polars as pl

LoadingAggregation = Literal["mean", "rms", "abs_mean"]
LOADING_AGGREGATIONS: tuple[LoadingAggregation, ...] = get_args(LoadingAggregation)


def aggregate_loadings_by_wavelength(
    components: pl.DataFrame, method: LoadingAggregation
) -> pl.DataFrame:
    """Aggregate component coefficients over the features of each wavelength.

    Parameters
    ----------
    components : pl.DataFrame
        Long-form coefficients with ``component``, ``wavelength``, and
        ``coefficient`` columns, such as the result of
        ``reshape_pca_components``. Other columns are ignored.
    method : LoadingAggregation
        ``"mean"`` for the mean coefficient, ``"rms"`` for the root mean
        square, or ``"abs_mean"`` for the mean absolute value. The mean lets
        coefficients of opposite signs cancel out; the others do not.

    Returns
    -------
    pl.DataFrame
        ``component``, ``wavelength``, and ``loading`` (``Float64``), one
        row per component and wavelength, sorted by both.

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
        for column in ("component", "wavelength", "coefficient")
        if column not in components.columns
    ]
    if missing:
        raise ValueError(f"components lack columns {missing}")
    return (
        components.group_by("component", "wavelength")
        .agg(expressions[method].alias("loading"))
        .sort("component", "wavelength")
    )
