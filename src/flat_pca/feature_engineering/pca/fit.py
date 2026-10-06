"""PCA fitting and transformation over the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from ..outlier import OutlierBounds, OutlierModel, OutlierStrategy, fit_outlier
from ..scaling import ScalingStrategy, fit_scaler
from .impute import ImputeModel, ImputeStrategy, fit_impute
from .mahalanobis import (
    MahalanobisConfig,
    mahalanobis_score_columns,
    validate_mahalanobis_request,
)
from .missing_rows import collect_complete_rows, select_rows
from .model import PcaModel
from .spe import SpeConfig, spe_score_columns, validate_spe_request


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
    if scaling_strategy not in {"none", "z-score", "minmax", "robust", "pareto"}:
        raise ValueError(
            "scaling_strategy must be 'none', 'z-score', 'minmax', 'robust', "
            "or 'pareto'"
        )

    _validate_columns(df, columns)


def _drops_missing_rows_in_numpy(
    impute_strategy: ImputeStrategy,
    outlier_strategy: OutlierStrategy,
    scaling_strategy: ScalingStrategy,
) -> bool:
    """Tell whether missing rows can be dropped on the collected matrix.

    With no outlier handling and no scaling, dropping rows is the only
    preprocessing step, so it can run on the NumPy matrix after a single
    ``select``/``collect`` instead of as a per-column Polars filter.

    Parameters
    ----------
    impute_strategy : {"drop", "median", "kmeans"}
        Missing-value handling strategy.
    outlier_strategy : OutlierStrategy
        Outlier handling performed before scaling.
    scaling_strategy : ScalingStrategy
        Scaling applied before PCA.

    Returns
    -------
    bool
        ``True`` for ``"drop"`` with no outlier handling and ``"none"``
        scaling.
    """
    return (
        impute_strategy == "drop"
        and outlier_strategy is None
        and scaling_strategy == "none"
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

    standardized_values: np.ndarray
    if _drops_missing_rows_in_numpy(
        impute_strategy, outlier_strategy, scaling_strategy
    ):
        # Every stage holds no fitted values here, so build their state
        # directly instead of the per-column drop predicate.
        standardized_values, _ = collect_complete_rows(df, columns)
        impute_model = ImputeModel(
            strategy=impute_strategy,
            values={},
            kmeans_n_clusters=None,
            kmeans_centroids=[],
        )
        outlier_model = OutlierModel(
            strategy=outlier_strategy,
            iqr_multiplier=iqr_multiplier,
            bounds=OutlierBounds.empty(),
        )
        scaling_model = fit_scaler(df, columns, scaling_strategy)
    else:
        # Each stage is fitted on the output of the previous one.
        prepared_frame, impute_model = fit_impute(
            df, columns, impute_strategy, impute_kmeans_n_clusters
        )
        prepared_frame, outlier_model = fit_outlier(
            prepared_frame, columns, outlier_strategy, iqr_multiplier
        )
        scaling_model = fit_scaler(prepared_frame, columns, scaling_strategy)
        standardized_values = (
            scaling_model.apply(prepared_frame, columns)
            .select(columns)
            .collect()
            .to_numpy()
        )

    n_samples = standardized_values.shape[0]
    if n_samples == 0:
        raise ValueError("no rows remain after preprocessing")

    if n_component is not None and n_component > n_samples:
        raise ValueError(
            "n_component must be less than or equal to the number of samples"
        )

    upper_bound = (
        len(columns)
        if max_n_component is None
        else min(len(columns), max_n_component)
    )
    fitted_n_component = (
        min(upper_bound, n_samples)
        if n_component is None
        else min(n_component, upper_bound)
    )

    pca = PCA(n_components=fitted_n_component)
    pca.fit(standardized_values)

    return PcaModel(
        columns=tuple(columns),
        n_component=fitted_n_component,
        impute_model=impute_model,
        outlier_model=outlier_model,
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
    *,
    mahalanobis: MahalanobisConfig | None = None,
    spe: SpeConfig | None = None,
) -> pl.LazyFrame:
    """Apply a fitted PCA pipeline and append score columns.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the model's feature columns.
    pca_model : PcaModel
        Fitted preprocessing and PCA state.
    mahalanobis : MahalanobisConfig | None, default None
        If given, append the squared Mahalanobis distance in PCA score
        space, its upper control limit, and the exceeds-limit flag after the
        score columns. ``None`` leaves the output unchanged.
    spe : SpeConfig | None, default None
        If given, append the Q statistic (squared reconstruction error of
        the preprocessed features from the leading components), its upper
        control limit, and the exceeds-limit flag after the score and
        Mahalanobis columns. ``None`` leaves the output unchanged.

    Returns
    -------
    pl.LazyFrame
        Prepared input columns with PCA score columns appended, followed by
        the Mahalanobis columns when ``mahalanobis`` is given and the SPE
        columns when ``spe`` is given.

    Raises
    ------
    ValueError
        If input data are incompatible, no rows remain after preparation, or
        the Mahalanobis or SPE request is invalid for the model or collides
        with existing columns.
    """
    columns = list(pca_model.columns)

    # The fitted model already carries validated settings, so only the input
    # frame's columns still need checking here.
    _validate_columns(df, columns)
    existing_columns = [*df.collect_schema().names(), *pca_model.pca_column_names]
    if mahalanobis is not None:
        validate_mahalanobis_request(pca_model.pca, mahalanobis, existing_columns)
        existing_columns.extend(mahalanobis.column_names)
    if spe is not None:
        validate_spe_request(pca_model.pca, spe, existing_columns)

    standardized_values: np.ndarray
    if _drops_missing_rows_in_numpy(
        pca_model.impute_model.strategy,
        pca_model.outlier_model.strategy,
        pca_model.scaling_model.strategy,
    ):
        standardized_values, complete_mask = collect_complete_rows(df, columns)
        scaled_frame = select_rows(df, complete_mask)
    else:
        # Apply the fitted stages in the order they were fitted.
        scaled_frame = df
        for stage in (
            pca_model.impute_model,
            pca_model.outlier_model,
            pca_model.scaling_model,
        ):
            scaled_frame = stage.apply(scaled_frame, columns)
        standardized_values = scaled_frame.select(columns).collect().to_numpy()

    if standardized_values.shape[0] == 0:
        raise ValueError("no rows remain after preprocessing")

    scores = pca_model.pca.transform(standardized_values)
    scores_frame = pl.DataFrame(
        {
            "__row_id": list(range(scores.shape[0])),
            **{
                pca_column_name: scores[:, index]
                for index, pca_column_name in enumerate(
                    pca_model.pca_column_names
                )
            },
            **(
                mahalanobis_score_columns(pca_model.pca, scores, mahalanobis)
                if mahalanobis is not None
                else {}
            ),
            **(
                spe_score_columns(pca_model.pca, standardized_values, scores, spe)
                if spe is not None
                else {}
            ),
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
    mahalanobis: MahalanobisConfig | None = None,
    spe: SpeConfig | None = None,
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
    mahalanobis : MahalanobisConfig | None, default None
        Mahalanobis distance settings forwarded to ``transform_pca``.
        ``None`` leaves the output unchanged.
    spe : SpeConfig | None, default None
        Q statistic settings forwarded to ``transform_pca``. ``None`` leaves
        the output unchanged.

    Returns
    -------
    pl.LazyFrame
        Prepared input columns with fitted PCA score columns appended,
        followed by the Mahalanobis columns when ``mahalanobis`` is given and
        the SPE columns when ``spe`` is given.

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
    return transform_pca(df, pca_model, mahalanobis=mahalanobis, spe=spe)
