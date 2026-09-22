"""PCA fitting and score appending for flattened spectral features."""

from __future__ import annotations

from numbers import Integral

import polars as pl

from flat_pca.spectral.schema import flattened_feature_columns

from ..pca import ImputeStrategy, PcaModel, fit_pca, transform_pca


def fit_flattened_pca(
    flattened: pl.LazyFrame,
    n_component: int | None,
    impute_strategy: ImputeStrategy = "drop",
    impute_kmeans_n_clusters: int | None = None,
) -> PcaModel:
    """Fit PCA from the non-source columns of flattened features.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened spectral features with a ``source`` column.
    n_component : int | None
        Requested number of components, or ``None`` for the matrix limit.
    impute_strategy : {"drop", "median", "kmeans"}, default "drop"
        Missing-value handling strategy forwarded to ``fit_pca``. ``"drop"``
        discards any row (input file) with a null or NaN in any feature
        column; ``"median"`` fills nulls/NaNs with each column's median
        instead; ``"kmeans"`` fills them from the nearest cluster centroid
        fitted on rows with no missing values. Applies to whatever nulls
        remain after any upstream column pruning (see
        ``drop_sparse_feature_columns``).
    impute_kmeans_n_clusters : int | None, default None
        Requested cluster count for ``impute_strategy="kmeans"``, forwarded
        to ``fit_pca``. Must be ``None`` for any other strategy.

    Returns
    -------
    PcaModel
        Fitted PCA pipeline state.

    Raises
    ------
    ValueError
        If ``n_component`` is not an integer in the permitted range.
    """
    feature_columns = flattened_feature_columns(flattened.collect_schema().names())
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
        impute_strategy=impute_strategy,
        impute_kmeans_n_clusters=impute_kmeans_n_clusters,
        outlier_strategy=None,
        scaling_strategy="none",
    )


def append_pca_scores(
    pca_model: PcaModel,
    flattened: pl.LazyFrame,
) -> pl.LazyFrame:
    """Append PCA scores to flattened features, applying the fitted impute strategy.

    Delegates to ``transform_pca`` so that any missing values remaining in
    ``flattened`` (for example from ``flatten_inputs`` filling a file's
    missing ``(Step, Sequence, StepTime)`` combination with null) are handled
    the same way they were at fit time, instead of being passed directly into
    ``sklearn``'s ``PCA.transform``, which rejects null/NaN input.

    Parameters
    ----------
    pca_model : PcaModel
        Fitted PCA pipeline state.
    flattened : pl.LazyFrame
        Flattened spectral features with a ``source`` column.

    Returns
    -------
    pl.LazyFrame
        ``source``, flattened feature columns, and ``pca-1`` onward scores.
        Feature-column values reflect ``pca_model.impute_strategy``: unchanged
        for ``"drop"`` (rows with remaining missing values are excluded
        instead), or filled with the fitted median for ``"median"``.

    Raises
    ------
    ValueError
        If the PCA feature count differs from the flattened feature count, or
        if no rows remain after missing-value handling.
    """
    feature_columns = flattened_feature_columns(flattened.collect_schema().names())
    if pca_model.pca.n_features_in_ != len(feature_columns):
        raise ValueError(
            "PCA feature count does not match flattened feature count"
        )

    return transform_pca(flattened, pca_model)
