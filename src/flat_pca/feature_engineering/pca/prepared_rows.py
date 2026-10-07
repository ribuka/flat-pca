"""Preparation of feature rows with a fitted model's missing-value handling.

Rows are imputed and outlier-handled with the fitted state, in the same
order as ``transform_pca``, so reconstructions and residuals can be computed
from complete values in the original feature scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .model import PcaModel

_ROW_INDEX = "__row"


@dataclass(frozen=True)
class PreparedRows:
    """Feature rows after the fitted imputation and outlier handling.

    Attributes
    ----------
    kept : np.ndarray
        Zero-based indices of the input rows kept by the imputation; rows
        with a missing value are dropped by ``impute_strategy="drop"``.
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

    The stages are applied as ``impute_model.apply`` then
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
    columns = list(model.columns)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns):
        raise ValueError(
            f"values are shaped {matrix.shape}, expected (n_rows, {len(columns)})"
        )
    frame = pl.from_numpy(matrix, schema=columns, orient="row").with_row_index(
        _ROW_INDEX
    )
    prepared = model.outlier_model.apply(
        model.impute_model.apply(frame.lazy(), columns), columns
    ).collect()
    scaled = model.scaling_model.apply(prepared.lazy(), columns).select(columns).collect()
    kept = prepared[_ROW_INDEX].to_numpy().astype(np.intp)
    if kept.size == 0:
        scores = np.empty((0, model.n_component), dtype=np.float64)
    else:
        scores = np.asarray(model.pca.transform(scaled.to_numpy()), dtype=np.float64)
    return PreparedRows(
        kept=kept,
        values=prepared.select(columns).to_numpy().astype(np.float64),
        scores=scores,
    )
