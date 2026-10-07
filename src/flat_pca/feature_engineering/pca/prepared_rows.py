"""Preparation of feature rows with a fitted model's missing-value handling.

Rows are imputed and outlier-handled with the fitted state, in the same
order as ``transform_pca``, so reconstructions and residuals can be computed
from complete values in the original feature scale. The stages run as NumPy
array operations over ``model.feature_arrays``, so preparing a few rows does
not cost one Polars expression per feature column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feature_arrays import handle_outlier_rows, impute_rows
from .model import PcaModel


@dataclass(frozen=True)
class PreparedRows:
    """Feature rows after the fitted imputation and outlier handling.

    Attributes
    ----------
    kept : np.ndarray
        Zero-based indices of the input rows kept by the imputation and the
        outlier handling; rows with a missing value are dropped by
        ``impute_strategy="drop"``, and rows with an outlier by
        ``outlier_strategy="drop"``.
    values : np.ndarray
        ``float64`` values of the kept rows, shaped
        ``(len(kept), n_features)``, imputed and outlier-handled but not
        scaled. They contain no missing value.
    scores : np.ndarray
        PCA scores of the kept rows, shaped ``(len(kept), n_component)``.
    """

    kept: np.ndarray
    values: np.ndarray
    scores: np.ndarray


def prepare_rows(values: np.ndarray, model: PcaModel) -> PreparedRows:
    """Impute, outlier-handle, and score feature rows with a fitted model.

    The stages agree with ``impute_model.apply`` then
    ``outlier_model.apply``, as in ``transform_pca``; the scores come from
    the scaled values, so they agree with ``transform_pca``.

    Parameters
    ----------
    values : np.ndarray
        Feature rows shaped ``(n_rows, len(model.columns))`` in the model's
        column order. NaN marks a missing value.
    model : PcaModel
        Fitted preprocessing and PCA state.

    Returns
    -------
    PreparedRows
        The kept rows, their prepared values, and their scores.

    Raises
    ------
    ValueError
        If ``values`` is not two-dimensional with one column per feature.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(model.columns):
        raise ValueError(
            f"values are shaped {matrix.shape}, expected (n_rows, {len(model.columns)})"
        )
    arrays = model.feature_arrays
    imputed_mask, imputed = impute_rows(matrix, arrays)
    outlier_mask, prepared = handle_outlier_rows(imputed, arrays)
    kept = np.flatnonzero(imputed_mask)[outlier_mask].astype(np.intp)
    if kept.size == 0:
        scores = np.empty((0, model.n_component), dtype=np.float64)
    else:
        scores = np.asarray(
            model.pca.transform(arrays.scaling.scale(prepared)), dtype=np.float64
        )
    return PreparedRows(kept=kept, values=prepared, scores=scores)


def scale_rows(values: np.ndarray, model: PcaModel) -> np.ndarray:
    """Scale prepared feature rows with the fitted scaling.

    Parameters
    ----------
    values : np.ndarray
        Rows shaped ``(n_rows, len(model.columns))`` in the model's column
        order, already imputed and outlier-handled with the model
        (``PreparedRows.values``).
    model : PcaModel
        Fitted preprocessing and PCA state.

    Returns
    -------
    np.ndarray
        ``float64`` rows after the fitted scaling, the values passed to
        ``pca.transform``.
    """
    return model.feature_arrays.scaling.scale(np.asarray(values, dtype=np.float64))
