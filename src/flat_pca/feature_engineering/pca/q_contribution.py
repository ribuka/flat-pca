"""Per-feature contributions to the Q statistic (SPE).

``Q = sum_f (x_f - x_hat_f)**2`` sums one squared residual per feature, in
the scaled space passed to ``pca.transform``. Keeping the terms apart shows
which features make a row's Q large.
"""

from __future__ import annotations

import numpy as np

from .mahalanobis import resolve_mahalanobis_components
from .model import PcaModel
from .prepared_rows import PreparedRows, scale_rows
from .reconstruct import reconstruct_standardized


def q_contribution(
    prepared: PreparedRows,
    model: PcaModel,
    cumulative_explained_variance: float | None,
) -> np.ndarray:
    """Return the squared residual of every feature of prepared rows.

    The prepared values are scaled with ``model.scaling_model.apply`` into
    the ``x`` passed to ``pca.transform``, and ``x_hat`` is their
    reconstruction from the leading components chosen by
    ``cumulative_explained_variance``, without undoing the scaling. The sum
    over the features of a row is the row's Q statistic computed with the
    same selector.

    Parameters
    ----------
    prepared : PreparedRows
        Rows prepared by ``prepare_rows`` with ``model``.
    model : PcaModel
        Fitted preprocessing and PCA state.
    cumulative_explained_variance : float | int | None
        Component selector of the Q statistic
        (``SpeConfig.cumulative_explained_variance``); see
        ``resolve_used_components``.

    Returns
    -------
    np.ndarray
        ``float64`` values ``(x_f - x_hat_f)**2`` shaped
        ``(len(prepared.kept), len(model.columns))``.

    Raises
    ------
    ValueError
        If the selector is out of range.
    """
    used_components = resolve_mahalanobis_components(
        model.pca, cumulative_explained_variance
    )
    scaled = scale_rows(prepared.values, model)
    return (scaled - reconstruct_standardized(prepared.scores, model.pca, used_components)) ** 2
