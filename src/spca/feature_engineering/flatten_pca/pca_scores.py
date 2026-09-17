"""PCA fitting and score appending for flattened spectral features."""

from __future__ import annotations

from numbers import Integral

import polars as pl
from sklearn.decomposition import PCA


def fit_flattened_pca(
    flattened: pl.LazyFrame,
    n_component: int | None,
) -> PCA:
    """Fit PCA from the non-filename columns of flattened features.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened spectral features with a ``filename`` column.
    n_component : int | None
        Requested number of components, or ``None`` for the matrix limit.

    Returns
    -------
    PCA
        Fitted scikit-learn PCA estimator.

    Raises
    ------
    ValueError
        If ``n_component`` is not an integer in the permitted range.
    """
    materialized = flattened.collect()
    feature_columns = [
        column for column in materialized.columns if column != "filename"
    ]
    max_component = min(materialized.height, len(feature_columns))
    if n_component is None:
        resolved_n_component = max_component
    elif (
        isinstance(n_component, bool)
        or not isinstance(n_component, Integral)
        or not 1 <= n_component <= max_component
    ):
        raise ValueError(
            f"n_component must be an integer between 1 and {max_component}"
        )
    else:
        resolved_n_component = int(n_component)

    pca = PCA(n_components=resolved_n_component)
    pca.fit(materialized.select(feature_columns).to_numpy())
    return pca


def append_pca_scores(
    pca: PCA,
    flattened: pl.LazyFrame,
) -> pl.LazyFrame:
    """Append PCA scores to flattened features without changing row order.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
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
    if pca.n_features_in_ != len(feature_columns):
        raise ValueError(
            "PCA feature count does not match flattened feature count"
        )

    scores = pca.transform(materialized.select(feature_columns).to_numpy())
    score_columns = {
        f"pca-{index + 1}": scores[:, index]
        for index in range(scores.shape[1])
    }
    return materialized.with_columns(
        [pl.Series(name, values) for name, values in score_columns.items()]
    ).lazy()
