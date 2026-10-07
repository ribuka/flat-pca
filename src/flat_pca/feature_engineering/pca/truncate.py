"""Truncating a fitted PCA pipeline to its leading components."""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.decomposition import PCA

from .model import PcaModel


def truncate_pca_model(model: PcaModel, n_component: int) -> PcaModel:
    """Keep only the leading components of a fitted PCA pipeline.

    The variance of the dropped components is folded into
    ``pca.noise_variance_``, the mean variance of the components that are
    not kept, so the total variance seen by the Q statistic's upper control
    limit is unchanged. The individual dropped variances are lost, which
    makes that limit less accurate (see ``spe.spe_ucl``).

    Parameters
    ----------
    model : PcaModel
        Fitted PCA pipeline.
    n_component : int
        Number of leading components to keep, between 1 and the fitted
        component count.

    Returns
    -------
    PcaModel
        ``model`` itself when ``n_component`` equals the fitted count,
        otherwise a copy holding the leading ``n_component`` components.

    Raises
    ------
    ValueError
        If ``n_component`` is outside ``1..`` the fitted component count.
    """
    pca = model.pca
    fitted = int(pca.n_components_)
    if isinstance(n_component, bool) or not 1 <= n_component <= fitted:
        raise ValueError(f"n_component must be between 1 and {fitted}")
    if n_component == fitted:
        return model

    rank = min(int(pca.n_samples_), int(pca.n_features_in_))
    explained_variance = np.asarray(pca.explained_variance_, dtype=float)
    unfitted_before = max(rank - fitted, 0)
    noise_variance = (
        float(np.sum(explained_variance[n_component:]))
        + unfitted_before * float(pca.noise_variance_)
    ) / (rank - n_component)

    truncated = PCA(n_components=n_component, whiten=bool(pca.whiten))
    truncated.components_ = pca.components_[:n_component].copy()
    truncated.mean_ = pca.mean_
    truncated.explained_variance_ = explained_variance[:n_component].copy()
    truncated.explained_variance_ratio_ = np.asarray(
        pca.explained_variance_ratio_[:n_component], dtype=float
    ).copy()
    truncated.singular_values_ = np.asarray(
        pca.singular_values_[:n_component], dtype=float
    ).copy()
    truncated.n_features_in_ = int(pca.n_features_in_)
    truncated.n_samples_ = int(pca.n_samples_)
    truncated.noise_variance_ = noise_variance
    truncated.n_components_ = n_component
    return dataclasses.replace(
        model,
        n_component=n_component,
        pca=truncated,
        pca_column_names=model.pca_column_names[:n_component],
    )
