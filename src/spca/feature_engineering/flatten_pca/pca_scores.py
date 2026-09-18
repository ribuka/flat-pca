"""PCA fitting and score appending for flattened spectral features."""

from __future__ import annotations

from numbers import Integral

import polars as pl

from ..pca import PcaModel, fit_pca


def fit_flattened_pca(
    flattened: pl.LazyFrame,
    n_component: int | None,
) -> PcaModel:
    """Fit PCA from the non-filename columns of flattened features.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened spectral features with a ``filename`` column.
    n_component : int | None
        Requested number of components, or ``None`` for the matrix limit.

    Returns
    -------
    PcaModel
        Fitted PCA pipeline state.

    Raises
    ------
    ValueError
        If ``n_component`` is not an integer in the permitted range.
    """
    feature_columns = [
        column
        for column in flattened.collect_schema().names()
        if column != "filename"
    ]
    if isinstance(n_component, bool) or (
        n_component is not None and not isinstance(n_component, Integral)
    ):
        raise ValueError("n_component must be an integer or None")
    if n_component is not None and n_component <= 0:
        raise ValueError("n_component must be a positive integer")
    if n_component is not None and n_component > len(feature_columns):
        raise ValueError(
            "n_component must be less than or equal to the flattened feature count"
        )
    return fit_pca(
        df=flattened,
        columns=feature_columns,
        n_component=n_component,
        max_n_component=None,
        impute_strategy="drop",
        outlier_strategy=None,
        scaling_strategy="none",
    )


def append_pca_scores(
    pca_model: PcaModel,
    flattened: pl.LazyFrame,
) -> pl.LazyFrame:
    """Append PCA scores to flattened features without changing row order.

    Parameters
    ----------
    pca_model : PcaModel
        Fitted PCA pipeline state.
    flattened : pl.LazyFrame
        Flattened spectral features with a ``filename`` column.

    Returns
    -------
    pl.LazyFrame
        ``filename``, flattened feature columns, and ``pca-1`` onward scores.

    Raises
    ------
    ValueError
        If the PCA feature count differs from the flattened feature count.
    """
    materialized = flattened.collect()
    feature_columns = [
        column for column in materialized.columns if column != "filename"
    ]
    if pca_model.pca.n_features_in_ != len(feature_columns):
        raise ValueError(
            "PCA feature count does not match flattened feature count"
        )

    scores = pca_model.pca.transform(materialized.select(feature_columns).to_numpy())
    score_columns = {
        f"pca-{index + 1}": scores[:, index]
        for index in range(scores.shape[1])
    }
    return materialized.with_columns(
        [pl.Series(name, values) for name, values in score_columns.items()]
    ).lazy()
