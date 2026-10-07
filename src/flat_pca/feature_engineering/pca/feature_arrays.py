"""Fitted per-feature preprocessing state as arrays in the model's column order.

The fitted imputation, outlier, and scaling models keep their per-feature
values in column-keyed mappings, which suit serialization and Polars
expressions over many rows. Applying them to a few rows through Polars costs
one expression per feature column, so the same state is laid out here as
NumPy arrays once per model and applied with array operations. The results
agree with ``ImputeModel.apply``, ``OutlierModel.apply``, and
``ScalingModel.apply``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..outlier import OutlierModel, OutlierStrategy
from ..scaling import ScalingModel
from .impute import ImputeModel, ImputeStrategy, nearest_centroid_fill


@dataclass(frozen=True)
class ScalingArrays:
    """Fitted feature scaling as arrays.

    Attributes
    ----------
    identity : bool
        ``True`` for ``strategy="none"``, which leaves values unchanged.
    centers : np.ndarray
        ``float64`` value subtracted from each feature before scaling.
    scales : np.ndarray
        ``float64`` nonzero divisor of each feature.
    """

    identity: bool
    centers: np.ndarray
    scales: np.ndarray

    def scale(self, values: np.ndarray) -> np.ndarray:
        """Apply the scaling, ``(value - center) / scale``.

        Parameters
        ----------
        values : np.ndarray
            Values of shape ``(n_rows, n_features)``.

        Returns
        -------
        np.ndarray
            Scaled values; ``values`` itself when the scaling is the identity.
        """
        if self.identity:
            return values
        return (values - self.centers) / self.scales

    def unscale(self, values: np.ndarray, *, center: bool = True) -> np.ndarray:
        """Undo the scaling, ``value * scale + center``.

        Parameters
        ----------
        values : np.ndarray
            Scaled values of shape ``(n_rows, n_features)``.
        center : bool, default True
            Whether to add the center. ``False`` only multiplies by the
            scale, which maps a difference of scaled values to the original
            scale.

        Returns
        -------
        np.ndarray
            Values in the original feature scale; ``values`` itself when the
            scaling is the identity.
        """
        if self.identity:
            return values
        if not center:
            return values * self.scales
        return values * self.scales + self.centers


@dataclass(frozen=True)
class FeatureArrays:
    """Fitted imputation, outlier, and scaling state as arrays.

    Every per-feature array follows the model's column order.

    Attributes
    ----------
    impute_strategy : {"drop", "median", "kmeans"}
        Missing-value handling strategy.
    impute_values : np.ndarray
        Median fill value of each feature; empty unless
        ``impute_strategy`` is ``"median"``.
    kmeans_centroids : np.ndarray
        Cluster centroids shaped ``(n_clusters, n_features)``; empty unless
        ``impute_strategy`` is ``"kmeans"``.
    outlier_strategy : {"winsorize", "drop"} | None
        Outlier-handling strategy.
    outlier_lower, outlier_upper : np.ndarray
        Outlier thresholds of each feature; empty when ``outlier_strategy``
        is ``None``.
    winsor_lower, winsor_upper : np.ndarray
        Clipping bounds of each feature; empty when ``outlier_strategy`` is
        ``None``.
    scaling : ScalingArrays
        Fitted feature scaling.
    """

    impute_strategy: ImputeStrategy
    impute_values: np.ndarray
    kmeans_centroids: np.ndarray
    outlier_strategy: OutlierStrategy
    outlier_lower: np.ndarray
    outlier_upper: np.ndarray
    winsor_lower: np.ndarray
    winsor_upper: np.ndarray
    scaling: ScalingArrays


def _ordered(values: Mapping[str, float], columns: Sequence[str], name: str) -> np.ndarray:
    """Lay out a column-keyed mapping as an array in ``columns`` order.

    Parameters
    ----------
    values : Mapping[str, float]
        Per-column values.
    columns : Sequence[str]
        Feature columns defining the output order.
    name : str
        Description of ``values`` used in the error message.

    Returns
    -------
    np.ndarray
        ``float64`` array of length ``len(columns)``.

    Raises
    ------
    ValueError
        If ``values`` lacks a column.
    """
    missing_columns = [column for column in columns if column not in values]
    if missing_columns:
        raise ValueError(f"{name} has no value for columns: {missing_columns}")
    return np.fromiter(
        (values[column] for column in columns), dtype=np.float64, count=len(columns)
    )


def build_scaling_arrays(
    scaling_model: ScalingModel, columns: Sequence[str]
) -> ScalingArrays:
    """Lay out fitted scaling state as arrays.

    Parameters
    ----------
    scaling_model : ScalingModel
        Fitted scaling state.
    columns : Sequence[str]
        Feature columns defining the array order.

    Returns
    -------
    ScalingArrays
        The centers and scales; the identity for ``strategy="none"``, whose
        mappings are not read.

    Raises
    ------
    ValueError
        If the scaling model lacks the center or scale of a column.
    """
    if scaling_model.strategy == "none":
        return ScalingArrays(
            identity=True,
            centers=np.zeros(len(columns)),
            scales=np.ones(len(columns)),
        )
    return ScalingArrays(
        identity=False,
        centers=_ordered(scaling_model.centers, columns, "scaling model centers"),
        scales=_ordered(scaling_model.scales, columns, "scaling model scales"),
    )


def build_feature_arrays(
    columns: Sequence[str],
    impute_model: ImputeModel,
    outlier_model: OutlierModel,
    scaling_model: ScalingModel,
) -> FeatureArrays:
    """Lay out fitted preprocessing state as arrays.

    Parameters
    ----------
    columns : Sequence[str]
        Feature columns defining the array order.
    impute_model : ImputeModel
        Fitted missing-value handling state.
    outlier_model : OutlierModel
        Fitted outlier-handling state.
    scaling_model : ScalingModel
        Fitted scaling state.

    Returns
    -------
    FeatureArrays
        The state of the three stages; only the mappings a strategy uses
        are read.

    Raises
    ------
    ValueError
        If a used mapping lacks a column.
    """
    empty = np.empty(0, dtype=np.float64)
    impute_values = (
        _ordered(impute_model.values, columns, "median imputation")
        if impute_model.strategy == "median"
        else empty
    )
    kmeans_centroids = (
        np.stack(
            [
                _ordered(centroid, columns, "kmeans centroid")
                for centroid in impute_model.kmeans_centroids
            ]
        )
        if impute_model.strategy == "kmeans"
        else np.empty((0, len(columns)), dtype=np.float64)
    )
    bounds = outlier_model.bounds
    handles_outliers = outlier_model.strategy is not None
    return FeatureArrays(
        impute_strategy=impute_model.strategy,
        impute_values=impute_values,
        kmeans_centroids=kmeans_centroids,
        outlier_strategy=outlier_model.strategy,
        outlier_lower=(
            _ordered(bounds.outlier_lower, columns, "outlier lower bounds")
            if handles_outliers
            else empty
        ),
        outlier_upper=(
            _ordered(bounds.outlier_upper, columns, "outlier upper bounds")
            if handles_outliers
            else empty
        ),
        winsor_lower=(
            _ordered(bounds.winsor_lower, columns, "winsor lower bounds")
            if handles_outliers
            else empty
        ),
        winsor_upper=(
            _ordered(bounds.winsor_upper, columns, "winsor upper bounds")
            if handles_outliers
            else empty
        ),
        scaling=build_scaling_arrays(scaling_model, columns),
    )


def impute_rows(
    values: np.ndarray, arrays: FeatureArrays
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fitted missing-value handling to feature rows.

    Parameters
    ----------
    values : np.ndarray
        ``float64`` rows shaped ``(n_rows, n_features)``; NaN marks a
        missing value.
    arrays : FeatureArrays
        Fitted preprocessing state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean mask of the kept rows (``"drop"`` removes rows with a NaN),
        and the kept rows with every missing value filled.
    """
    missing = np.isnan(values)
    if arrays.impute_strategy == "drop":
        kept = ~missing.any(axis=1)
        return kept, values[kept]
    kept = np.ones(values.shape[0], dtype=bool)
    if arrays.impute_strategy == "kmeans":
        return kept, nearest_centroid_fill(values, arrays.kmeans_centroids)
    return kept, np.where(missing, arrays.impute_values, values)


def handle_outlier_rows(
    values: np.ndarray, arrays: FeatureArrays
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fitted outlier handling to imputed feature rows.

    Parameters
    ----------
    values : np.ndarray
        ``float64`` rows shaped ``(n_rows, n_features)`` with no missing
        value.
    arrays : FeatureArrays
        Fitted preprocessing state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean mask of the kept rows (``"drop"`` removes rows with an
        outlier), and the kept rows; ``"winsorize"`` clips every value of a
        row with an outlier to the clipping bounds.
    """
    kept = np.ones(values.shape[0], dtype=bool)
    if arrays.outlier_strategy is None:
        return kept, values
    is_outlier = (
        (values < arrays.outlier_lower) | (values > arrays.outlier_upper)
    ).any(axis=1)
    if arrays.outlier_strategy == "drop":
        return ~is_outlier, values[~is_outlier]
    clipped = np.minimum(np.maximum(values, arrays.winsor_lower), arrays.winsor_upper)
    return kept, np.where(is_outlier[:, np.newaxis], clipped, values)
