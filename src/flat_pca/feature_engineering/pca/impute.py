"""K-means based missing-value imputation for PCA feature columns.

Complements the ``"drop"``/``"median"`` handling in ``fit.py`` with a
``"kmeans"`` option that fills each missing feature value from the nearest
cluster centroid fitted on rows with no missing values, instead of a single
global per-column statistic.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

ImputeStrategy = Literal["drop", "median", "kmeans"]

DEFAULT_KMEANS_N_CLUSTERS = 8
_KMEANS_RANDOM_STATE = 0


def resolve_kmeans_n_clusters(n_clusters: int | None, complete_row_count: int) -> int:
    """Resolve the requested cluster count against the available complete rows.

    Parameters
    ----------
    n_clusters : int | None
        Requested cluster count, or ``None`` to use ``DEFAULT_KMEANS_N_CLUSTERS``.
    complete_row_count : int
        Number of rows with no missing value in any feature column.

    Returns
    -------
    int
        Cluster count clipped to at most ``complete_row_count``.

    Raises
    ------
    ValueError
        If ``complete_row_count`` is 0.
    """
    if complete_row_count == 0:
        raise ValueError(
            "kmeans imputation requires at least one row without missing values"
        )
    resolved = DEFAULT_KMEANS_N_CLUSTERS if n_clusters is None else n_clusters
    return min(resolved, complete_row_count)


def _nearest_centroid_fill(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Fill each row's missing entries from its nearest cluster centroid.

    Parameters
    ----------
    values : np.ndarray
        Feature values shaped ``(n_rows, n_features)``. Missing entries are
        ``NaN``.
    centroids : np.ndarray
        Cluster centroids shaped ``(n_clusters, n_features)``.

    Returns
    -------
    np.ndarray
        ``values`` with every ``NaN`` replaced by its row's nearest
        centroid value, using only the row's non-missing dimensions to pick
        the nearest centroid. Non-missing entries are unchanged.
    """
    missing = np.isnan(values)
    present = ~missing
    zero_filled = np.where(missing, 0.0, values)
    diff = (zero_filled[:, None, :] - centroids[None, :, :]) * present[:, None, :]
    distances = np.square(diff).sum(axis=2)
    nearest = distances.argmin(axis=1)
    return np.where(missing, centroids[nearest], values)


def _centroids_to_payload(
    centroids: np.ndarray, columns: list[str]
) -> list[dict[str, float]]:
    """Convert fitted centroids into a JSON-serializable per-cluster mapping.

    Parameters
    ----------
    centroids : np.ndarray
        Cluster centroids shaped ``(n_clusters, n_features)``.
    columns : list[str]
        Feature column names matching ``centroids``' column order.

    Returns
    -------
    list[dict[str, float]]
        One column-to-value mapping per cluster centroid.
    """
    return [
        {column: float(value) for column, value in zip(columns, center, strict=True)}
        for center in centroids
    ]


def _centroids_from_payload(
    centroids: list[dict[str, float]], columns: list[str]
) -> np.ndarray:
    """Convert a per-cluster mapping back into a centroid array.

    Parameters
    ----------
    centroids : list[dict[str, float]]
        Previously fitted centroids, as produced by ``_centroids_to_payload``.
    columns : list[str]
        Feature column names, defining the output column order.

    Returns
    -------
    np.ndarray
        Cluster centroids shaped ``(n_clusters, n_features)``.
    """
    return np.array(
        [[centroid[column] for column in columns] for centroid in centroids],
        dtype=float,
    )


def fit_kmeans_impute(
    df: pl.LazyFrame,
    columns: list[str],
    n_clusters: int | None,
) -> tuple[pl.LazyFrame, int, list[dict[str, float]]]:
    """Fit cluster centroids on complete rows and fill missing values.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to impute.
    n_clusters : int | None
        Requested cluster count, or ``None`` to use
        ``DEFAULT_KMEANS_N_CLUSTERS``. Clipped to the number of rows with no
        missing value in ``columns``.

    Returns
    -------
    tuple[pl.LazyFrame, int, list[dict[str, float]]]
        Data with missing values filled, the resolved cluster count, and the
        fitted centroids (one column-to-value mapping per cluster).

    Raises
    ------
    ValueError
        If no row is free of missing values in ``columns``.
    """
    collected = df.collect()
    values = collected.select(columns).to_numpy().astype(float)
    complete_row_mask = ~np.isnan(values).any(axis=1)
    resolved_n_clusters = resolve_kmeans_n_clusters(
        n_clusters, int(complete_row_mask.sum())
    )

    kmeans = KMeans(
        n_clusters=resolved_n_clusters,
        random_state=_KMEANS_RANDOM_STATE,
        n_init="auto",
    )
    kmeans.fit(values[complete_row_mask])
    centroids = kmeans.cluster_centers_

    filled = _nearest_centroid_fill(values, centroids)
    prepared = collected.with_columns(
        [pl.Series(column, filled[:, index]) for index, column in enumerate(columns)]
    )
    return (
        prepared.lazy(),
        resolved_n_clusters,
        _centroids_to_payload(centroids, columns),
    )


def apply_kmeans_impute(
    df: pl.LazyFrame,
    columns: list[str],
    centroids: list[dict[str, float]],
) -> pl.LazyFrame:
    """Fill missing values using previously fitted cluster centroids.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to impute.
    centroids : list[dict[str, float]]
        Cluster centroids fitted by ``fit_kmeans_impute``.

    Returns
    -------
    pl.LazyFrame
        Data with missing values filled from their row's nearest fitted
        centroid.
    """
    collected = df.collect()
    values = collected.select(columns).to_numpy().astype(float)
    filled = _nearest_centroid_fill(values, _centroids_from_payload(centroids, columns))
    prepared = collected.with_columns(
        [pl.Series(column, filled[:, index]) for index, column in enumerate(columns)]
    )
    return prepared.lazy()
