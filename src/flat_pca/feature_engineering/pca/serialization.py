"""JSON-compatible serialization of a fitted PCA pipeline.

``parse_transform_payload`` returns ``PcaModel`` constructor arguments
rather than a ``PcaModel`` itself, so this module never imports the model
container at run time and the two modules stay free of a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.decomposition import PCA

from ..outlier import OutlierModel
from ..scaling import ScalingModel
from .impute import ImputeModel

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
        **model.impute_model.to_payload(),
        **model.outlier_model.to_payload(),
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
        "impute_model": ImputeModel.from_payload(payload),
        "outlier_model": OutlierModel.from_payload(payload),
        "scaling_model": ScalingModel.from_payload(
            cast(dict[str, object], payload["scaling_model"])
        ),
        "pca": _restore_pca(cast(dict[str, object], payload["pca"]), n_component),
        "pca_column_names": tuple(cast(list[str], payload["pca_column_names"])),
    }
