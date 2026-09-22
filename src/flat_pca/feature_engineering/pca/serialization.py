"""JSON-compatible serialization of a fitted PCA pipeline.

``parse_transform_payload`` returns ``PcaModel`` constructor arguments
rather than a ``PcaModel`` itself, so this module never imports the model
container at run time and the two modules stay free of a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from sklearn.decomposition import PCA

from ..outlier import OutlierStrategy
from ..scaling import ScalingModel, ScalingStrategy

if TYPE_CHECKING:
    from .model import PcaModel


def _float_map(
    payload: dict[str, object],
    key: str,
    *,
    required: bool = False,
) -> dict[str, float]:
    """Read one payload entry as a column-to-float mapping.

    Parameters
    ----------
    payload : dict[str, object]
        Serialized state to read from.
    key : str
        Entry name.
    required : bool, default False
        If ``True``, a missing entry raises ``KeyError``. If ``False``, a
        missing entry is treated as an empty mapping, which keeps payloads
        written before the optional entries existed readable.

    Returns
    -------
    dict[str, float]
        Per-column floating-point values.

    Raises
    ------
    KeyError
        If ``required`` is ``True`` and the entry is missing.
    """
    raw = cast(
        dict[str, float],
        payload[key] if required else payload.get(key, {}),
    )
    return {column: float(value) for column, value in raw.items()}


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
        "impute_strategy": model.impute_strategy,
        "impute_values": model.impute_values,
        "outlier_strategy": model.outlier_strategy,
        "iqr_multiplier": model.iqr_multiplier,
        "outlier_lower_bounds": model.outlier_lower_bounds,
        "outlier_upper_bounds": model.outlier_upper_bounds,
        "winsor_lower_bounds": model.winsor_lower_bounds,
        "winsor_upper_bounds": model.winsor_upper_bounds,
        "scaling_model": {
            "strategy": model.scaling_model.strategy,
            "centers": model.scaling_model.centers,
            "scales": model.scaling_model.scales,
        },
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


def _restore_scaling_model(payload: dict[str, object]) -> ScalingModel:
    """Rebuild fitted feature-scaling state from its serialized form.

    Parameters
    ----------
    payload : dict[str, object]
        ``scaling_model`` entry of a transform payload.

    Returns
    -------
    ScalingModel
        Restored scaling state.
    """
    return ScalingModel(
        strategy=cast(ScalingStrategy, payload["strategy"]),
        centers=_float_map(payload, "centers", required=True),
        scales=_float_map(payload, "scales", required=True),
    )


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

    raw_outlier_strategy = payload.get("outlier_strategy", None)
    if raw_outlier_strategy == "none":
        raw_outlier_strategy = None

    return {
        "columns": tuple(cast(list[str], payload["columns"])),
        "n_component": n_component,
        "impute_strategy": cast(
            Literal["drop", "median"],
            payload["impute_strategy"],
        ),
        "impute_values": _float_map(payload, "impute_values", required=True),
        "outlier_strategy": cast(OutlierStrategy, raw_outlier_strategy),
        "iqr_multiplier": float(payload.get("iqr_multiplier", 1.5)),
        "outlier_lower_bounds": _float_map(payload, "outlier_lower_bounds"),
        "outlier_upper_bounds": _float_map(payload, "outlier_upper_bounds"),
        "winsor_lower_bounds": _float_map(payload, "winsor_lower_bounds"),
        "winsor_upper_bounds": _float_map(payload, "winsor_upper_bounds"),
        "scaling_model": _restore_scaling_model(
            cast(dict[str, object], payload["scaling_model"])
        ),
        "pca": _restore_pca(cast(dict[str, object], payload["pca"]), n_component),
        "pca_column_names": tuple(cast(list[str], payload["pca_column_names"])),
    }
