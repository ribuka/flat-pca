"""PCA fitting and transformation over the preprocessing pipeline."""

from __future__ import annotations

from typing import Literal

import polars as pl
from sklearn.decomposition import PCA

from ..outlier import OutlierBounds, OutlierStrategy, prepare_outlier_frame
from ..scaling import ScalingStrategy, apply_scaler, fit_scaler
from .impute import ImputeStrategy, apply_kmeans_impute, fit_kmeans_impute
from .model import PcaModel


def _validate_columns(df: pl.LazyFrame, columns: list[str]) -> None:
    """Check that the requested feature columns exist in the input.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data to inspect.
    columns : list[str]
        Feature columns the caller expects to be present.

    Raises
    ------
    ValueError
        If ``columns`` is empty or names a column the input lacks.
    """
    if not columns:
        raise ValueError("columns must not be empty")
    schema_names = set(df.collect_schema().names())
    missing_columns = [column for column in columns if column not in schema_names]
    if missing_columns:
        raise ValueError(f"columns not found in df: {missing_columns}")


def _validate_fit_args(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None,
    max_n_component: int | None,
    impute_strategy: ImputeStrategy,
    impute_kmeans_n_clusters: int | None,
    outlier_strategy: OutlierStrategy,
    iqr_multiplier: float,
    scaling_strategy: ScalingStrategy,
) -> None:
    """Validate every ``fit_pca`` argument before any work is done.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected feature columns.
    columns : list[str]
        Numeric columns used to fit PCA.
    n_component : int | None
        Requested component count, or ``None``.
    max_n_component : int | None
        Additional component-count cap, or ``None``.
    impute_strategy : {"drop", "median", "kmeans"}
        Missing-value handling strategy.
    impute_kmeans_n_clusters : int | None
        Requested cluster count for ``impute_strategy="kmeans"``, or
        ``None``. Must be ``None`` for any other strategy.
    outlier_strategy : OutlierStrategy
        Outlier handling performed before scaling.
    iqr_multiplier : float
        Multiplier used to calculate IQR outlier bounds.
    scaling_strategy : ScalingStrategy
        Scaling applied before PCA fitting.

    Raises
    ------
    ValueError
        If any argument is outside its permitted range or set of values, or
        if a requested column is missing from ``df``.
    """
    if not columns:
        raise ValueError("columns must not be empty")
    if n_component is not None and n_component <= 0:
        raise ValueError("n_component must be greater than 0")
    if max_n_component is not None:
        if max_n_component <= 0:
            raise ValueError("max_n_component must be greater than 0")
        if n_component is not None and n_component > max_n_component:
            raise ValueError("n_component must be less than or equal to max_n_component")
    if impute_strategy not in {"drop", "median", "kmeans"}:
        raise ValueError("impute_strategy must be 'drop', 'median', or 'kmeans'")
    if impute_kmeans_n_clusters is not None:
        if impute_strategy != "kmeans":
            raise ValueError(
                "impute_kmeans_n_clusters must be None unless "
                "impute_strategy is 'kmeans'"
            )
        if (
            isinstance(impute_kmeans_n_clusters, bool)
            or not isinstance(impute_kmeans_n_clusters, int)
            or impute_kmeans_n_clusters <= 0
        ):
            raise ValueError("impute_kmeans_n_clusters must be a positive integer")
    if outlier_strategy not in {None, "winsorize", "drop"}:
        raise ValueError("outlier_strategy must be None, 'winsorize', or 'drop'")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than 0")
    if scaling_strategy not in {"none", "z-score", "minmax", "robust"}:
        raise ValueError(
            "scaling_strategy must be 'none', 'z-score', 'minmax', or 'robust'"
        )

    _validate_columns(df, columns)


def _prepare_input_frame(
    df: pl.LazyFrame,
    columns: list[str],
    impute_strategy: Literal["drop", "median"],
    impute_values: dict[str, float] | None = None,
) -> tuple[pl.LazyFrame, dict[str, float]]:
    """Apply the configured missing-value handling to the feature columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric feature columns to handle.
    impute_strategy : {"drop", "median"}
        ``"drop"`` removes rows with any null or NaN feature; ``"median"``
        fills them with a per-column median.
    impute_values : dict[str, float] | None, default None
        Previously fitted medians to reuse. If ``None``, medians are
        computed from ``df`` itself.

    Returns
    -------
    tuple[pl.LazyFrame, dict[str, float]]
        Prepared data and the imputation values that were applied, which are
        empty for ``impute_strategy="drop"``.
    """
    if impute_strategy == "drop":
        missing_exprs = [
            pl.col(column).is_null() | pl.col(column).is_nan()
            for column in columns
        ]
        return df.filter(~pl.any_horizontal(missing_exprs)), {}

    resolved_impute_values = impute_values
    if resolved_impute_values is None:
        medians = (
            df.select([pl.col(column).median().alias(column) for column in columns])
            .collect()
            .row(0, named=True)
        )
        resolved_impute_values = {
            column: float(medians[column]) if medians[column] is not None else 0.0
            for column in columns
        }

    return (
        df.with_columns(
            [
                pl.col(column)
                .fill_null(resolved_impute_values[column])
                .fill_nan(resolved_impute_values[column])
                .alias(column)
                for column in columns
            ]
        ),
        resolved_impute_values,
    )


def fit_pca(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None = None,
    max_n_component: int | None = 100,
    impute_strategy: ImputeStrategy = "drop",
    outlier_strategy: OutlierStrategy = None,
    iqr_multiplier: float = 1.5,
    scaling_strategy: ScalingStrategy = "robust",
    *,
    impute_kmeans_n_clusters: int | None = None,
) -> PcaModel:
    """Fit a PCA model with the configured preprocessing pipeline.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected feature columns.
    columns : list[str]
        Numeric columns used to fit PCA.
    n_component : int | None, default None
        Requested component count, or ``None`` to use the configured maximum.
    max_n_component : int | None, default 100
        Additional component-count cap, or ``None`` for no cap.
    impute_strategy : {"drop", "median", "kmeans"}, default "drop"
        Missing-value handling strategy. ``"kmeans"`` fills each missing
        value from the nearest cluster centroid fitted on rows with no
        missing values; see ``impute.fit_kmeans_impute``.
    outlier_strategy : OutlierStrategy, default None
        Optional outlier handling performed before scaling.
    iqr_multiplier : float, default 1.5
        Positive multiplier used to calculate IQR outlier bounds.
    scaling_strategy : ScalingStrategy, default "robust"
        Scaling applied before PCA fitting.
    impute_kmeans_n_clusters : int | None, default None
        Requested cluster count for ``impute_strategy="kmeans"``, or
        ``None`` to use ``impute.DEFAULT_KMEANS_N_CLUSTERS``. Must be
        ``None`` for any other strategy. Keyword-only so it can be added
        without disturbing existing positional ``fit_pca`` calls.

    Returns
    -------
    PcaModel
        Fitted preprocessing and PCA state.

    Raises
    ------
    ValueError
        If arguments, columns, or the prepared data are invalid.
    """
    _validate_fit_args(
        df,
        columns,
        n_component,
        max_n_component,
        impute_strategy,
        impute_kmeans_n_clusters,
        outlier_strategy,
        iqr_multiplier,
        scaling_strategy,
    )

    impute_kmeans_fitted_n_clusters: int | None = None
    impute_kmeans_centroids: list[dict[str, float]] = []
    if impute_strategy == "kmeans":
        prepared_frame, impute_kmeans_fitted_n_clusters, impute_kmeans_centroids = (
            fit_kmeans_impute(df, columns, impute_kmeans_n_clusters)
        )
        impute_values: dict[str, float] = {}
    else:
        prepared_frame, impute_values = _prepare_input_frame(
            df,
            columns,
            impute_strategy,
        )
    prepared_frame, outlier_bounds = prepare_outlier_frame(
        prepared_frame,
        columns,
        outlier_strategy,
        iqr_multiplier,
    )

    scaling_model = fit_scaler(
        prepared_frame,
        columns,
        scaling_strategy,
    )
    scaled_frame = apply_scaler(
        prepared_frame,
        columns,
        scaling_model,
    )

    standardized_frame = scaled_frame.select(columns).collect()
    if standardized_frame.height == 0:
        raise ValueError("no rows remain after preprocessing")

    if n_component is not None and n_component > standardized_frame.height:
        raise ValueError(
            "n_component must be less than or equal to the number of samples"
        )

    upper_bound = (
        len(columns)
        if max_n_component is None
        else min(len(columns), max_n_component)
    )
    fitted_n_component = (
        min(upper_bound, standardized_frame.height)
        if n_component is None
        else min(n_component, upper_bound)
    )

    pca = PCA(n_components=fitted_n_component)
    pca.fit(standardized_frame.to_numpy())

    return PcaModel(
        columns=tuple(columns),
        n_component=fitted_n_component,
        impute_strategy=impute_strategy,
        impute_values=impute_values,
        impute_kmeans_n_clusters=impute_kmeans_fitted_n_clusters,
        impute_kmeans_centroids=impute_kmeans_centroids,
        outlier_strategy=outlier_strategy,
        iqr_multiplier=iqr_multiplier,
        outlier_lower_bounds=outlier_bounds.outlier_lower,
        outlier_upper_bounds=outlier_bounds.outlier_upper,
        winsor_lower_bounds=outlier_bounds.winsor_lower,
        winsor_upper_bounds=outlier_bounds.winsor_upper,
        scaling_model=scaling_model,
        pca=pca,
        pca_column_names=tuple(
            f"pca-{i + 1}"
            for i in range(fitted_n_component)
        ),
    )


def transform_pca(
    df: pl.LazyFrame,
    pca_model: PcaModel,
) -> pl.LazyFrame:
    """Apply a fitted PCA pipeline and append score columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the model's feature columns.
    pca_model : PcaModel
        Fitted preprocessing and PCA state.

    Returns
    -------
    pl.LazyFrame
        Prepared input columns with PCA score columns appended.

    Raises
    ------
    ValueError
        If input data are incompatible or no rows remain after preparation.
    """
    columns = list(pca_model.columns)

    # The fitted model already carries validated settings, so only the input
    # frame's columns still need checking here.
    _validate_columns(df, columns)

    # Missing-value handling.
    if pca_model.impute_strategy == "kmeans":
        prepared_frame = apply_kmeans_impute(
            df,
            columns,
            pca_model.impute_kmeans_centroids,
        )
    else:
        prepared_frame, _ = _prepare_input_frame(
            df,
            columns,
            pca_model.impute_strategy,
            pca_model.impute_values,
        )

    # Outlier handling, reusing the thresholds fitted with the model.
    prepared_frame, _ = prepare_outlier_frame(
        prepared_frame,
        columns,
        pca_model.outlier_strategy,
        pca_model.iqr_multiplier,
        OutlierBounds(
            outlier_lower=pca_model.outlier_lower_bounds,
            outlier_upper=pca_model.outlier_upper_bounds,
            winsor_lower=pca_model.winsor_lower_bounds,
            winsor_upper=pca_model.winsor_upper_bounds,
        ),
    )

    # Feature scaling.
    scaled_frame = apply_scaler(
        prepared_frame,
        columns,
        pca_model.scaling_model,
    )

    standardized_frame = scaled_frame.select(columns).collect()
    if standardized_frame.height == 0:
        raise ValueError("no rows remain after preprocessing")

    scores = pca_model.pca.transform(standardized_frame.to_numpy())
    scores_frame = pl.DataFrame(
        {
            "__row_id": list(range(scores.shape[0])),
            **{
                pca_column_name: scores[:, index]
                for index, pca_column_name in enumerate(
                    pca_model.pca_column_names
                )
            },
        }
    )

    return (
        scaled_frame.with_row_index("__row_id")
        .join(scores_frame.lazy(), on="__row_id", how="left")
        .drop("__row_id")
    )


def fit_and_transform_pca(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None = None,
    max_n_component: int | None = 100,
    impute_strategy: ImputeStrategy = "drop",
    outlier_strategy: OutlierStrategy = None,
    iqr_multiplier: float = 1.5,
    scaling_strategy: ScalingStrategy = "robust",
    *,
    impute_kmeans_n_clusters: int | None = None,
) -> pl.LazyFrame:
    """Fit and apply PCA in one call.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected feature columns.
    columns : list[str]
        Numeric columns used to fit and transform PCA.
    n_component : int | None, default None
        Requested component count, or ``None`` to use the configured maximum.
    max_n_component : int | None, default 100
        Additional component-count cap, or ``None`` for no cap.
    impute_strategy : {"drop", "median", "kmeans"}, default "drop"
        Missing-value handling strategy.
    outlier_strategy : OutlierStrategy, default None
        Optional outlier handling performed before scaling.
    iqr_multiplier : float, default 1.5
        Positive multiplier used to calculate IQR outlier bounds.
    scaling_strategy : ScalingStrategy, default "robust"
        Scaling applied before PCA fitting.
    impute_kmeans_n_clusters : int | None, default None
        Requested cluster count for ``impute_strategy="kmeans"``, or
        ``None`` to use ``impute.DEFAULT_KMEANS_N_CLUSTERS``. Must be
        ``None`` for any other strategy. Keyword-only so it can be added
        without disturbing existing positional ``fit_and_transform_pca``
        calls.

    Returns
    -------
    pl.LazyFrame
        Prepared input columns with fitted PCA score columns appended.

    Raises
    ------
    ValueError
        If arguments, columns, or the prepared data are invalid.
    """
    pca_model = fit_pca(
        df=df,
        columns=columns,
        n_component=n_component,
        max_n_component=max_n_component,
        impute_strategy=impute_strategy,
        impute_kmeans_n_clusters=impute_kmeans_n_clusters,
        outlier_strategy=outlier_strategy,
        iqr_multiplier=iqr_multiplier,
        scaling_strategy=scaling_strategy,
    )
    return transform_pca(df, pca_model)
