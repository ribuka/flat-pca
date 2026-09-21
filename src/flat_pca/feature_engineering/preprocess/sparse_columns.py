"""Sparse feature-column pruning for flattened spectral features."""

from __future__ import annotations

import polars as pl

from ...utils.pl_snippets import (
    drop_all_null_columns_from_polars,
    validate_missing_ratio_threshold,
)


def validate_max_null_ratio(max_null_ratio: float) -> None:
    """Validate the public sparse-feature missing-value threshold.

    Parameters
    ----------
    max_null_ratio : float
        Inclusive maximum missing-value ratio for retained features.

    Raises
    ------
    ValueError
        If ``max_null_ratio`` is not between 0.0 and 1.0.
    """
    validate_missing_ratio_threshold(max_null_ratio)


def drop_sparse_feature_columns(
    flattened: pl.LazyFrame,
    max_null_ratio: float,
) -> pl.LazyFrame:
    """Drop flattened feature columns whose missing-value ratio is too high.

    Different input files may cover different (Step, Sequence, StepTime)
    combinations, so ``flatten_inputs`` fills a file's missing combination
    with null across every wavelength at that combination. Left unpruned,
    PCA fitting's default ``impute_strategy="drop"`` drops any row with a
    null in any feature column, which can discard every row. Pruning
    columns that are mostly missing keeps the columns actually shared
    across inputs, so rows survive that row-wise imputation.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened features with a ``source`` column plus one feature column
        per (wavelength, Step, Sequence, StepTime) combination.
    max_null_ratio : float
        Inclusive upper bound (0.0-1.0) on a column's null-or-NaN ratio for
        it to be kept. Use 1.0 to disable pruning (only a fully null/NaN
        column would still be dropped).

    Returns
    -------
    pl.LazyFrame
        ``flattened`` restricted to columns whose missing ratio is at most
        ``max_null_ratio``. ``source`` has a 0.0 ratio and is always kept.

    Raises
    ------
    ValueError
        If ``max_null_ratio`` is not between 0.0 and 1.0.
    """
    validate_max_null_ratio(max_null_ratio)
    return drop_all_null_columns_from_polars(
        flattened, include_nan_missing=True, threshold=max_null_ratio
    )
