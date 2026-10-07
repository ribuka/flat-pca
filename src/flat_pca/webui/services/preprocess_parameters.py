"""Per-feature preprocessing parameters of a fitted PCA model shown as heatmaps.

A ``PcaModel`` keeps one value per feature for each fitted preprocessing
stage: the centering mean of the PCA, the center and scale of the scaling,
the fill values of the imputation, and the thresholds of the outlier
handling. Which of them exist depends on the run's strategies, so the model
screen offers only the ones the run has.
"""

from __future__ import annotations

import re

import numpy as np

from flat_pca.feature_engineering.pca import PcaModel

FIXED_PARAMETER_LABELS: dict[str, str] = {
    "mean": "中心化の平均",
    "scaling_center": "スケーリングの center",
    "scaling_scale": "スケーリングの scale",
    "impute_median": "補完値（中央値）",
    "outlier_lower": "外れ値の閾値（下限）",
    "outlier_upper": "外れ値の閾値（上限）",
    "winsor_lower": "winsorize の下限",
    "winsor_upper": "winsorize の上限",
}
PARAMETER_VALUE_NAME = "value"
_CENTROID_KEY = re.compile(r"kmeans_centroid_([1-9][0-9]*)")


def is_parameter_key(key: str) -> bool:
    """Return whether ``key`` names a preprocessing parameter of some run.

    Parameters
    ----------
    key : str
        Requested parameter key.

    Returns
    -------
    bool
        ``True`` for a key of ``FIXED_PARAMETER_LABELS`` or
        ``"kmeans_centroid_{c}"`` with a 1-based cluster number ``c``.
    """
    return key in FIXED_PARAMETER_LABELS or _CENTROID_KEY.fullmatch(key) is not None


def parameter_options(model: PcaModel) -> dict[str, str]:
    """Return the preprocessing parameters a model holds, with their labels.

    Parameters
    ----------
    model : PcaModel
        Fitted PCA model.

    Returns
    -------
    dict[str, str]
        Labels keyed by parameter key: the centering mean always; the
        scaling center and scale unless ``scaling_strategy="none"``; the
        median fill values for ``impute_strategy="median"``; one centroid
        per cluster for ``"kmeans"``; and the outlier thresholds and
        clipping bounds when outliers are handled.
    """
    keys = ["mean"]
    if model.scaling_model.strategy != "none":
        keys += ["scaling_center", "scaling_scale"]
    if model.impute_model.strategy == "median":
        keys.append("impute_median")
    options = {key: FIXED_PARAMETER_LABELS[key] for key in keys}
    if model.impute_model.strategy == "kmeans":
        for cluster in range(1, len(model.impute_model.kmeans_centroids) + 1):
            options[f"kmeans_centroid_{cluster}"] = f"補完値（kmeans 重心 {cluster}）"
    if model.outlier_model.strategy is not None:
        for key in ("outlier_lower", "outlier_upper", "winsor_lower", "winsor_upper"):
            options[key] = FIXED_PARAMETER_LABELS[key]
    return options


def parameter_values(model: PcaModel, key: str) -> np.ndarray:
    """Return one preprocessing parameter's value of each feature.

    Parameters
    ----------
    model : PcaModel
        Fitted PCA model.
    key : str
        A key of ``parameter_options(model)``.

    Returns
    -------
    np.ndarray
        ``float64`` values in ``model.columns`` order, which is the order of
        the run's ``features.parquet``.

    Raises
    ------
    ValueError
        If the model does not hold the parameter.
    """
    if key not in parameter_options(model):
        raise ValueError(f"the model holds no parameter {key!r}")
    arrays = model.feature_arrays
    centroid = _CENTROID_KEY.fullmatch(key)
    if centroid is not None:
        return arrays.kmeans_centroids[int(centroid.group(1)) - 1]
    by_key = {
        "mean": np.asarray(model.pca.mean_, dtype=np.float64),
        "scaling_center": arrays.scaling.centers,
        "scaling_scale": arrays.scaling.scales,
        "impute_median": arrays.impute_values,
        "outlier_lower": arrays.outlier_lower,
        "outlier_upper": arrays.outlier_upper,
        "winsor_lower": arrays.winsor_lower,
        "winsor_upper": arrays.winsor_upper,
    }
    return by_key[key]
