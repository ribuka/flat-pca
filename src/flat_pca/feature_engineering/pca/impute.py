"""Missing-value imputation for PCA feature columns.

``"drop"`` removes rows with any missing feature value, ``"median"`` fills
them with a per-column median, and ``"kmeans"`` fills each missing feature
value from the nearest cluster centroid fitted on rows with no missing
values, instead of a single global per-column statistic. ``ImputeModel``
holds the fitted state of whichever strategy was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from ..payload import read_float_map

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


def nearest_centroid_fill(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
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

    filled = nearest_centroid_fill(values, centroids)
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
    filled = nearest_centroid_fill(values, _centroids_from_payload(centroids, columns))
    prepared = collected.with_columns(
        [pl.Series(column, filled[:, index]) for index, column in enumerate(columns)]
    )
    return prepared.lazy()


def drop_missing_rows(df: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Remove rows with a null or NaN value in any feature column.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to inspect.

    Returns
    -------
    pl.LazyFrame
        Rows of ``df`` whose feature values are all present.
    """
    missing_exprs = [
        pl.col(column).is_null() | pl.col(column).is_nan()
        for column in columns
    ]
    return df.filter(~pl.any_horizontal(missing_exprs))


def fit_median_values(df: pl.LazyFrame, columns: list[str]) -> dict[str, float]:
    """Compute the per-column medians used for median imputation.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to summarize.

    Returns
    -------
    dict[str, float]
        Median of each column, or ``0.0`` for a column with no value.
    """
    medians = (
        df.select([pl.col(column).median().alias(column) for column in columns])
        .collect()
        .row(0, named=True)
    )
    return {
        column: float(medians[column]) if medians[column] is not None else 0.0
        for column in columns
    }


def apply_median_impute(
    df: pl.LazyFrame,
    columns: list[str],
    values: dict[str, float],
) -> pl.LazyFrame:
    """Fill null and NaN feature values with fitted per-column values.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to impute.
    values : dict[str, float]
        Per-column fill values fitted by ``fit_median_values``.

    Returns
    -------
    pl.LazyFrame
        Data with missing feature values filled.
    """
    return df.with_columns(
        [
            pl.col(column)
            .fill_null(values[column])
            .fill_nan(values[column])
            .alias(column)
            for column in columns
        ]
    )


def _read_kmeans_centroids(payload: dict[str, object]) -> list[dict[str, float]]:
    """Read the ``impute_kmeans_centroids`` entry as per-cluster float maps.

    Parameters
    ----------
    payload : dict[str, object]
        Serialized state to read from.

    Returns
    -------
    list[dict[str, float]]
        One column-to-float mapping per cluster centroid.

    Raises
    ------
    KeyError
        If the entry is missing.
    """
    raw = cast(list[dict[str, float]], payload["impute_kmeans_centroids"])
    return [
        {column: float(value) for column, value in centroid.items()}
        for centroid in raw
    ]


@dataclass(frozen=True)
class ImputeModel:
    """Store fitted missing-value imputation state.

    Attributes
    ----------
    strategy : {"drop", "median", "kmeans"}
        Missing-value handling strategy.
    values : dict[str, float]
        Per-column values used for median imputation, empty unless
        ``strategy`` is ``"median"``.
    kmeans_n_clusters : int | None
        Fitted cluster count used for kmeans imputation, or ``None`` unless
        ``strategy`` is ``"kmeans"``.
    kmeans_centroids : list[dict[str, float]]
        Fitted cluster centroids used for kmeans imputation (one
        column-to-value mapping per cluster), empty unless ``strategy`` is
        ``"kmeans"``.
    """

    strategy: ImputeStrategy
    values: dict[str, float]
    kmeans_n_clusters: int | None
    kmeans_centroids: list[dict[str, float]]

    def apply(self, df: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
        """Apply the fitted missing-value handling to feature columns.

        Parameters
        ----------
        df : pl.LazyFrame
            Input data containing ``columns``.
        columns : list[str]
            Numeric feature columns to handle.

        Returns
        -------
        pl.LazyFrame
            Data with missing rows dropped or missing values filled.
        """
        if self.strategy == "drop":
            return drop_missing_rows(df, columns)
        if self.strategy == "kmeans":
            return apply_kmeans_impute(df, columns, self.kmeans_centroids)
        return apply_median_impute(df, columns, self.values)

    def to_payload(self) -> dict[str, object]:
        """Return the fitted imputation state in JSON-compatible form.

        Returns
        -------
        dict[str, object]
            ``impute_strategy``, ``impute_values``,
            ``impute_kmeans_n_clusters``, and ``impute_kmeans_centroids``
            entries.
        """
        return {
            "impute_strategy": self.strategy,
            "impute_values": self.values,
            "impute_kmeans_n_clusters": self.kmeans_n_clusters,
            "impute_kmeans_centroids": self.kmeans_centroids,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ImputeModel:
        """Restore fitted imputation state from its serialized form.

        Parameters
        ----------
        payload : dict[str, object]
            State previously produced by :meth:`to_payload`.
            ``impute_kmeans_n_clusters`` is ``None`` unless the strategy is
            ``"kmeans"``.

        Returns
        -------
        ImputeModel
            Restored imputation state.

        Raises
        ------
        KeyError
            If an entry is missing.
        """
        raw_n_clusters = payload["impute_kmeans_n_clusters"]
        return cls(
            strategy=cast(ImputeStrategy, payload["impute_strategy"]),
            values=read_float_map(payload, "impute_values"),
            kmeans_n_clusters=(
                int(cast(int, raw_n_clusters)) if raw_n_clusters is not None else None
            ),
            kmeans_centroids=_read_kmeans_centroids(payload),
        )


def fit_impute(
    df: pl.LazyFrame,
    columns: list[str],
    strategy: ImputeStrategy,
    kmeans_n_clusters: int | None = None,
) -> tuple[pl.LazyFrame, ImputeModel]:
    """Fit missing-value handling on the data and apply it.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing ``columns``.
    columns : list[str]
        Numeric feature columns to handle.
    strategy : {"drop", "median", "kmeans"}
        Missing-value handling strategy. ``"drop"`` does not read the data.
    kmeans_n_clusters : int | None, default None
        Requested cluster count for ``strategy="kmeans"``, or ``None`` to
        use ``DEFAULT_KMEANS_N_CLUSTERS``. Ignored for other strategies.

    Returns
    -------
    tuple[pl.LazyFrame, ImputeModel]
        Handled data and the fitted imputation state.

    Raises
    ------
    ValueError
        If ``strategy`` is ``"kmeans"`` and no row is free of missing values.
    """
    if strategy == "kmeans":
        prepared, fitted_n_clusters, centroids = fit_kmeans_impute(
            df, columns, kmeans_n_clusters
        )
        return prepared, ImputeModel(
            strategy=strategy,
            values={},
            kmeans_n_clusters=fitted_n_clusters,
            kmeans_centroids=centroids,
        )

    values = fit_median_values(df, columns) if strategy == "median" else {}
    model = ImputeModel(
        strategy=strategy,
        values=values,
        kmeans_n_clusters=None,
        kmeans_centroids=[],
    )
    return model.apply(df, columns), model
