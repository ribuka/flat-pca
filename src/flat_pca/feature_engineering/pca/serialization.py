"""Serialization of a fitted PCA pipeline.

Two forms are supported: a JSON-compatible payload, and NumPy arrays (for
``.npy``/``.npz`` files) whose per-feature values follow the model's column
order. ``parse_transform_payload`` and ``parse_pca_state`` return
``PcaModel`` constructor arguments rather than a ``PcaModel`` itself, so
this module never imports the model container at run time and the two
modules stay free of a circular import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.decomposition import PCA

from ..outlier import OutlierBounds, OutlierModel, OutlierStrategy
from ..scaling import ScalingModel, ScalingStrategy
from .impute import ImputeModel, ImputeStrategy

if TYPE_CHECKING:
    from .model import PcaModel


def _float_array(payload: dict[str, object], key: str) -> np.ndarray:
    """Read one payload entry as a floating-point array.

    Parameters
    ----------
    payload : dict[str, object]
        Serialized state to read from.
    key : str
        Entry name.

    Returns
    -------
    np.ndarray
        Floating-point array built from the entry.
    """
    return np.asarray(payload[key], dtype=float)


def build_transform_payload(model: PcaModel) -> dict[str, object]:
    """Return parameters required to reproduce a model's transformation.

    Parameters
    ----------
    model : PcaModel
        Fitted preprocessing and PCA state.

    Returns
    -------
    dict[str, object]
        JSON-serializable preprocessing and PCA state.
    """
    return {
        "columns": list(model.columns),
        "n_component": model.n_component,
        "impute_model": model.impute_model.to_payload(),
        "outlier_model": model.outlier_model.to_payload(),
        "scaling_model": model.scaling_model.to_payload(),
        "pca_column_names": list(model.pca_column_names),
        "pca": {
            "components": model.pca.components_.tolist(),
            "mean": model.pca.mean_.tolist(),
            "explained_variance": model.pca.explained_variance_.tolist(),
            "explained_variance_ratio": model.pca.explained_variance_ratio_.tolist(),
            "singular_values": model.pca.singular_values_.tolist(),
            "n_features_in": int(model.pca.n_features_in_),
            "n_samples": int(model.pca.n_samples_),
            "noise_variance": float(model.pca.noise_variance_),
            "whiten": bool(model.pca.whiten),
        },
    }


def _restore_pca(payload: dict[str, object], n_component: int) -> PCA:
    """Rebuild a fitted scikit-learn estimator from its serialized attributes.

    Parameters
    ----------
    payload : dict[str, object]
        ``pca`` entry of a transform payload.
    n_component : int
        Number of fitted principal components.

    Returns
    -------
    PCA
        Estimator carrying the serialized fitted attributes.
    """
    pca = PCA(n_components=n_component, whiten=bool(payload["whiten"]))
    pca.components_ = _float_array(payload, "components")
    pca.mean_ = _float_array(payload, "mean")
    pca.explained_variance_ = _float_array(payload, "explained_variance")
    pca.explained_variance_ratio_ = _float_array(payload, "explained_variance_ratio")
    pca.singular_values_ = _float_array(payload, "singular_values")
    pca.n_features_in_ = int(payload["n_features_in"])
    pca.n_samples_ = int(payload["n_samples"])
    pca.noise_variance_ = float(payload["noise_variance"])
    pca.n_components_ = pca.components_.shape[0]
    return pca


def parse_transform_payload(payload: dict[str, object]) -> dict[str, object]:
    """Convert serialized state into ``PcaModel`` constructor arguments.

    Parameters
    ----------
    payload : dict[str, object]
        State previously produced by ``build_transform_payload``.

    Returns
    -------
    dict[str, object]
        Keyword arguments accepted by ``PcaModel``.
    """
    n_component = int(payload["n_component"])

    return {
        "columns": tuple(cast(list[str], payload["columns"])),
        "n_component": n_component,
        "impute_model": ImputeModel.from_payload(
            cast(dict[str, object], payload["impute_model"])
        ),
        "outlier_model": OutlierModel.from_payload(
            cast(dict[str, object], payload["outlier_model"])
        ),
        "scaling_model": ScalingModel.from_payload(
            cast(dict[str, object], payload["scaling_model"])
        ),
        "pca": _restore_pca(cast(dict[str, object], payload["pca"]), n_component),
        "pca_column_names": tuple(cast(list[str], payload["pca_column_names"])),
    }


# ``build_pca_state`` stores a ``None`` outlier strategy and kmeans cluster
# count as these values, since ``.npz`` arrays cannot hold ``None``.
_NO_OUTLIER_STRATEGY = ""
_NO_KMEANS_N_CLUSTERS = 0


def _feature_array(values: Mapping[str, float], columns: Sequence[str]) -> np.ndarray:
    """Order per-feature values by the model's columns.

    Parameters
    ----------
    values : Mapping[str, float]
        Values keyed by feature column; empty when the stage holds none.
    columns : Sequence[str]
        Feature columns of the model.

    Returns
    -------
    np.ndarray
        ``float64`` array of ``len(columns)`` values, or an empty array when
        ``values`` is empty.

    Raises
    ------
    ValueError
        If ``values`` is neither empty nor keyed by exactly ``columns``.
    """
    if not values:
        return np.empty(0, dtype=np.float64)
    if len(values) != len(columns):
        raise ValueError("per-feature values must cover every model column")
    return np.array([values[column] for column in columns], dtype=np.float64)


def _feature_map(
    state: Mapping[str, np.ndarray], name: str, columns: Sequence[str]
) -> dict[str, float]:
    """Key a saved per-feature array by the model's columns.

    Parameters
    ----------
    state : Mapping[str, np.ndarray]
        Saved state arrays.
    name : str
        Entry holding values in ``columns`` order, or an empty array.
    columns : Sequence[str]
        Feature columns of the model.

    Returns
    -------
    dict[str, float]
        Values keyed by column; empty for an empty array.

    Raises
    ------
    KeyError
        If the entry is missing.
    ValueError
        If the array is neither empty nor one value per column.
    """
    values = np.asarray(state[name], dtype=np.float64)
    if values.size == 0:
        return {}
    if values.shape != (len(columns),):
        raise ValueError(f"pca state {name!r} must hold one value per feature")
    return {column: float(value) for column, value in zip(columns, values, strict=True)}


def build_pca_state(model: PcaModel) -> dict[str, np.ndarray]:
    """Return a model's fitted state, except components, as NumPy arrays.

    The arrays can be saved with ``np.savez`` and read back without pickle.
    Per-feature values follow ``model.columns``; a stage that holds no
    values for its strategy stores an empty array, and so does ``"none"``
    scaling, whose centers are all 0 and scales all 1. The component matrix is
    left out so that it can be saved on its own as a ``.npy`` file.

    Parameters
    ----------
    model : PcaModel
        Fitted preprocessing and PCA state.

    Returns
    -------
    dict[str, np.ndarray]
        ``n_component`` and ``pca_*`` entries for the PCA estimator,
        ``impute_*`` for the imputation stage, ``outlier_*`` for the outlier
        stage, and ``scaling_*`` for the scaling stage.
    """
    columns = model.columns
    pca = model.pca
    impute = model.impute_model
    outlier = model.outlier_model
    scaling = model.scaling_model
    centroids = np.array(
        [[centroid[column] for column in columns] for centroid in impute.kmeans_centroids],
        dtype=np.float64,
    ).reshape(len(impute.kmeans_centroids), len(columns))
    return {
        "n_component": np.array(model.n_component),
        "pca_column_names": np.array(model.pca_column_names, dtype=np.str_),
        "pca_mean": np.asarray(pca.mean_, dtype=np.float64),
        "pca_explained_variance": np.asarray(pca.explained_variance_, dtype=np.float64),
        "pca_explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        "pca_singular_values": np.asarray(pca.singular_values_, dtype=np.float64),
        "pca_n_samples": np.array(int(pca.n_samples_)),
        "pca_noise_variance": np.array(float(pca.noise_variance_)),
        "pca_whiten": np.array(bool(pca.whiten)),
        "impute_strategy": np.array(impute.strategy),
        "impute_values": _feature_array(impute.values, columns),
        "impute_kmeans_n_clusters": np.array(
            _NO_KMEANS_N_CLUSTERS
            if impute.kmeans_n_clusters is None
            else impute.kmeans_n_clusters
        ),
        "impute_kmeans_centroids": centroids,
        "outlier_strategy": np.array(
            _NO_OUTLIER_STRATEGY if outlier.strategy is None else outlier.strategy
        ),
        "outlier_iqr_multiplier": np.array(float(outlier.iqr_multiplier)),
        "outlier_outlier_lower": _feature_array(outlier.bounds.outlier_lower, columns),
        "outlier_outlier_upper": _feature_array(outlier.bounds.outlier_upper, columns),
        "outlier_winsor_lower": _feature_array(outlier.bounds.winsor_lower, columns),
        "outlier_winsor_upper": _feature_array(outlier.bounds.winsor_upper, columns),
        "scaling_strategy": np.array(scaling.strategy),
        "scaling_centers": _feature_array(
            {} if scaling.strategy == "none" else scaling.centers, columns
        ),
        "scaling_scales": _feature_array(
            {} if scaling.strategy == "none" else scaling.scales, columns
        ),
    }


def _validate_pca_shapes(pca: PCA, n_features: int) -> None:
    """Check that restored PCA attributes fit the component and feature counts.

    Parameters
    ----------
    pca : PCA
        Restored estimator.
    n_features : int
        Number of feature columns.

    Raises
    ------
    ValueError
        If an attribute's shape does not match.
    """
    n_component = int(pca.n_components_)
    if pca.components_.shape != (n_component, n_features):
        raise ValueError(
            f"components must be shaped ({n_component}, {n_features}), "
            f"got {pca.components_.shape}"
        )
    if pca.mean_.shape != (n_features,):
        raise ValueError("pca state 'pca_mean' must hold one value per feature")
    for name in (
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
    ):
        if getattr(pca, name).shape != (n_component,):
            raise ValueError(f"pca state 'pca_{name[:-1]}' must hold one value per component")


def parse_pca_state(
    columns: Sequence[str],
    components: np.ndarray,
    state: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Convert arrays saved by ``build_pca_state`` into ``PcaModel`` arguments.

    Floating-point arrays are converted to ``float64``, so state saved in
    ``float32`` (such as a ``float32`` component matrix) is computed with in
    ``float64``.

    Parameters
    ----------
    columns : Sequence[str]
        Feature columns, in the order the arrays were saved in.
    components : np.ndarray
        Component matrix shaped ``(n_component, len(columns))``.
    state : Mapping[str, np.ndarray]
        Arrays produced by ``build_pca_state``, e.g. an opened ``.npz`` file.

    Returns
    -------
    dict[str, object]
        Keyword arguments accepted by ``PcaModel``.

    Raises
    ------
    KeyError
        If an entry is missing.
    ValueError
        If an array's shape does not match the columns or components.
    """
    columns = tuple(columns)
    n_component = int(state["n_component"])
    pca = _restore_pca(
        {
            "components": components,
            "mean": state["pca_mean"],
            "explained_variance": state["pca_explained_variance"],
            "explained_variance_ratio": state["pca_explained_variance_ratio"],
            "singular_values": state["pca_singular_values"],
            "n_features_in": len(columns),
            "n_samples": state["pca_n_samples"],
            "noise_variance": state["pca_noise_variance"],
            "whiten": state["pca_whiten"],
        },
        n_component,
    )
    _validate_pca_shapes(pca, len(columns))
    pca_column_names = tuple(str(name) for name in state["pca_column_names"])
    if len(pca_column_names) != n_component:
        raise ValueError("pca state 'pca_column_names' must name every component")

    raw_n_clusters = int(state["impute_kmeans_n_clusters"])
    centroids = np.asarray(state["impute_kmeans_centroids"], dtype=np.float64)
    if centroids.ndim != 2 or (centroids.size and centroids.shape[1] != len(columns)):
        raise ValueError(
            "pca state 'impute_kmeans_centroids' must be shaped (n_clusters, features)"
        )
    raw_outlier_strategy = str(state["outlier_strategy"])
    scaling_strategy = cast(ScalingStrategy, str(state["scaling_strategy"]))
    centers = _feature_map(state, "scaling_centers", columns)
    scales = _feature_map(state, "scaling_scales", columns)
    if scaling_strategy == "none":
        centers = {column: 0.0 for column in columns}
        scales = {column: 1.0 for column in columns}
    return {
        "columns": columns,
        "n_component": n_component,
        "impute_model": ImputeModel(
            strategy=cast(ImputeStrategy, str(state["impute_strategy"])),
            values=_feature_map(state, "impute_values", columns),
            kmeans_n_clusters=(
                None if raw_n_clusters == _NO_KMEANS_N_CLUSTERS else raw_n_clusters
            ),
            kmeans_centroids=[
                {column: float(value) for column, value in zip(columns, row, strict=True)}
                for row in centroids
            ],
        ),
        "outlier_model": OutlierModel(
            strategy=cast(
                OutlierStrategy,
                None
                if raw_outlier_strategy == _NO_OUTLIER_STRATEGY
                else raw_outlier_strategy,
            ),
            iqr_multiplier=float(state["outlier_iqr_multiplier"]),
            bounds=OutlierBounds(
                outlier_lower=_feature_map(state, "outlier_outlier_lower", columns),
                outlier_upper=_feature_map(state, "outlier_outlier_upper", columns),
                winsor_lower=_feature_map(state, "outlier_winsor_lower", columns),
                winsor_upper=_feature_map(state, "outlier_winsor_upper", columns),
            ),
        ),
        "scaling_model": ScalingModel(
            strategy=scaling_strategy, centers=centers, scales=scales
        ),
        "pca": pca,
        "pca_column_names": pca_column_names,
    }
