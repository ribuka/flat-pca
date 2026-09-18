from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from .outlier import OutlierStrategy, _prepare_outlier_dataframe
from .scaling import ScalingModel, ScalingStrategy, apply_scaler, fit_scaler


@dataclass(frozen=True)
class PcaModel:
    """Store a fitted PCA pipeline and its preprocessing state.

    Attributes
    ----------
    columns : tuple[str, ...]
        Feature columns expected by the model.
    n_component : int
        Number of fitted principal components.
    impute_strategy : {"drop", "median"}
        Missing-value handling strategy.
    impute_values : dict[str, float]
        Per-column values used for median imputation.
    outlier_strategy : OutlierStrategy
        Fitted outlier-handling strategy.
    iqr_multiplier : float
        IQR multiplier used to determine outlier bounds.
    outlier_lower_bounds : dict[str, float]
        Per-column lower outlier thresholds.
    outlier_upper_bounds : dict[str, float]
        Per-column upper outlier thresholds.
    winsor_lower_bounds : dict[str, float]
        Per-column lower clipping bounds.
    winsor_upper_bounds : dict[str, float]
        Per-column upper clipping bounds.
    scaling_model : ScalingModel
        Fitted feature-scaling state.
    pca : PCA
        Fitted scikit-learn PCA estimator.
    pca_column_names : tuple[str, ...]
        Output score-column names.
    """

    columns: tuple[str, ...]
    n_component: int
    impute_strategy: Literal["drop", "median"]
    impute_values: dict[str, float]
    outlier_strategy: OutlierStrategy
    iqr_multiplier: float
    outlier_lower_bounds: dict[str, float]
    outlier_upper_bounds: dict[str, float]
    winsor_lower_bounds: dict[str, float]
    winsor_upper_bounds: dict[str, float]
    scaling_model: ScalingModel
    pca: PCA
    pca_column_names: tuple[str, ...]

    def get_component_coefficients(self, component: int) -> pl.DataFrame:
        """Return coefficients for one component ordered by absolute value.

        Parameters
        ----------
        component : int
            One-based component number.

        Returns
        -------
        pl.DataFrame
            Feature coefficients sorted by descending absolute magnitude.

        Raises
        ------
        ValueError
            If the component number is outside the fitted component range.
        """
        if component <= 0:
            raise ValueError("component must be greater than 0")
        if component > self.n_component:
            raise ValueError("component must be less than or equal to n_component")

        component_index = component - 1
        coefficients = self.pca.components_[component_index]
        rows = [
            {
                "column": column,
                "coefficient": round(float(coefficients[index]), 4),
                "abs_coefficient": round(float(abs(coefficients[index])), 4),
            }
            for index, column in enumerate(self.columns)
        ]
        return pl.DataFrame(rows).sort("abs_coefficient", descending=True)

    def get_feature_contribution_ranking(
        self,
        cumulative_explained_variance: float | None = None,
        include_component_breakdown: bool = False,
        component_prefix: str = "contribution_pc",
    ) -> pl.DataFrame:
        """Return ranked feature contributions decomposed from explained variance.

        Parameters
        ----------
        cumulative_explained_variance : float | int | None, default None
            Component selector for leading components.
            If ``None``, all fitted components are used.
            If ``int`` (>=1), it is treated as a component-count cap.
            If ``float`` with ``0 < value <= 1``, it is treated as a cumulative
            explained-variance threshold.
        include_component_breakdown : bool, default False
            If ``True``, append per-component contribution columns such as
            ``contribution_pc1``, ``contribution_pc2``, and so on.
        component_prefix : str, default "contribution_pc"
            Prefix used when naming per-component contribution columns.

        Returns
        -------
        pl.DataFrame
            Ranking table sorted by ``contribution`` in descending order with columns:
            ``rank``, ``feature``, ``contribution``,
            ``used_components``, and ``used_cumulative_explained_variance``.
            When ``include_component_breakdown`` is ``True``, per-component
            contribution columns are included as well.

        Raises
        ------
        ValueError
            If ``cumulative_explained_variance`` is out of range, unsupported in type,
            or if the PCA state is inconsistent.
        """
        explained_variance_ratio = np.asarray(
            self.pca.explained_variance_ratio_,
            dtype=float,
        )
        component_matrix = np.asarray(self.pca.components_, dtype=float)
        component_count = component_matrix.shape[0]

        if explained_variance_ratio.shape[0] != component_count:
            raise ValueError(
                "inconsistent PCA state: explained_variance_ratio and components"
            )
        if component_matrix.shape[1] != len(self.columns):
            raise ValueError("inconsistent PCA state: components and feature columns")

        if cumulative_explained_variance is None:
            used_components = component_count
        elif isinstance(cumulative_explained_variance, bool):
            raise ValueError(
                "cumulative_explained_variance must be None, int>=1, or float in (0, 1]"
            )
        elif isinstance(cumulative_explained_variance, Integral):
            component_cap = int(cumulative_explained_variance)
            if component_cap < 1:
                raise ValueError(
                    "cumulative_explained_variance as int must be >= 1"
                )
            used_components = min(component_cap, component_count)
        elif isinstance(cumulative_explained_variance, Real):
            variance_threshold = float(cumulative_explained_variance)
            if variance_threshold <= 0 or variance_threshold > 1:
                raise ValueError(
                    "cumulative_explained_variance as float must satisfy 0 < value <= 1"
                )
            cumulative_ratio = np.cumsum(explained_variance_ratio)
            used_components = int(
                np.searchsorted(cumulative_ratio, variance_threshold, side="left") + 1
            )
            used_components = min(used_components, component_count)
        else:
            raise ValueError(
                "cumulative_explained_variance must be None, int>=1, or float in (0, 1]"
            )

        used_component_matrix = component_matrix[:used_components]
        used_explained_variance_ratio = explained_variance_ratio[:used_components]
        component_contribution_matrix = (
            (used_component_matrix**2) * used_explained_variance_ratio[:, np.newaxis]
        )
        contribution_by_feature = component_contribution_matrix.sum(axis=0)
        used_cumulative_explained_variance = float(
            used_explained_variance_ratio.sum()
        )

        frame_data: dict[str, object] = {
            "feature": list(self.columns),
            "contribution": contribution_by_feature,
        }
        if include_component_breakdown:
            contribution_by_feature_and_component = component_contribution_matrix.T
            frame_data.update(
                {
                    f"{component_prefix}{component_index + 1}": (
                        contribution_by_feature_and_component[:, component_index]
                    )
                    for component_index in range(used_components)
                }
            )

        ranking_df = (
            pl.DataFrame(frame_data)
            .sort("contribution", descending=True)
            .with_row_index("rank", offset=1)
            .with_columns(
                pl.lit(used_components).alias("used_components"),
                pl.lit(used_cumulative_explained_variance).alias(
                    "used_cumulative_explained_variance"
                ),
            )
        )
        return ranking_df

    def to_transform_payload(self) -> dict[str, object]:
        """Return parameters required to reproduce transformation.

        Returns
        -------
        dict[str, object]
            JSON-serializable preprocessing and PCA state.
        """
        return {
            "columns": list(self.columns),
            "n_component": self.n_component,
            "impute_strategy": self.impute_strategy,
            "impute_values": self.impute_values,
            "outlier_strategy": self.outlier_strategy,
            "iqr_multiplier": self.iqr_multiplier,
            "outlier_lower_bounds": self.outlier_lower_bounds,
            "outlier_upper_bounds": self.outlier_upper_bounds,
            "winsor_lower_bounds": self.winsor_lower_bounds,
            "winsor_upper_bounds": self.winsor_upper_bounds,
            "scaling_model": {
                "strategy": self.scaling_model.strategy,
                "centers": self.scaling_model.centers,
                "scales": self.scaling_model.scales,
            },
            "pca_column_names": list(self.pca_column_names),
            "pca": {
                "components": self.pca.components_.tolist(),
                "mean": self.pca.mean_.tolist(),
                "explained_variance": self.pca.explained_variance_.tolist(),
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "singular_values": self.pca.singular_values_.tolist(),
                "n_features_in": int(self.pca.n_features_in_),
                "n_samples": int(self.pca.n_samples_),
                "noise_variance": float(self.pca.noise_variance_),
                "whiten": bool(self.pca.whiten),
            },
        }

    def to_transform_json(self) -> str:
        """Serialize the transformation state as JSON.

        Returns
        -------
        str
            JSON representation of the preprocessing and PCA state.
        """
        return json.dumps(self.to_transform_payload(), ensure_ascii=False)

    @classmethod
    def from_transform_payload(cls, payload: dict[str, object]) -> PcaModel:
        """Restore a model from transformation payload data.

        Parameters
        ----------
        payload : dict[str, object]
            State previously produced by :meth:`to_transform_payload`.

        Returns
        -------
        PcaModel
            Reconstructed model ready for transformation.
        """
        columns = tuple(cast(list[str], payload["columns"]))
        n_component = int(payload["n_component"])
        impute_strategy = cast(
            Literal["drop", "median"],
            payload["impute_strategy"],
        )
        impute_values = {
            key: float(value)
            for key, value in cast(
                dict[str, float],
                payload["impute_values"],
            ).items()
        }

        raw_outlier_strategy = payload.get("outlier_strategy", None)
        if raw_outlier_strategy == "none":
            raw_outlier_strategy = None
        outlier_strategy = cast(OutlierStrategy, raw_outlier_strategy)
        iqr_multiplier = float(payload.get("iqr_multiplier", 1.5))
        outlier_lower_bounds = {
            key: float(value)
            for key, value in cast(
                dict[str, float],
                payload.get("outlier_lower_bounds", {}),
            ).items()
        }
        outlier_upper_bounds = {
            key: float(value)
            for key, value in cast(
                dict[str, float],
                payload.get("outlier_upper_bounds", {}),
            ).items()
        }
        winsor_lower_bounds = {
            key: float(value)
            for key, value in cast(
                dict[str, float],
                payload.get("winsor_lower_bounds", {}),
            ).items()
        }
        winsor_upper_bounds = {
            key: float(value)
            for key, value in cast(
                dict[str, float],
                payload.get("winsor_upper_bounds", {}),
            ).items()
        }

        scaling_payload = cast(
            dict[str, object],
            payload["scaling_model"],
        )
        scaling_model = ScalingModel(
            strategy=cast(ScalingStrategy, scaling_payload["strategy"]),
            centers={
                key: float(value)
                for key, value in cast(
                    dict[str, float],
                    scaling_payload["centers"],
                ).items()
            },
            scales={
                key: float(value)
                for key, value in cast(
                    dict[str, float],
                    scaling_payload["scales"],
                ).items()
            },
        )

        pca_payload = cast(dict[str, object], payload["pca"])
        pca = PCA(
            n_components=n_component,
            whiten=bool(pca_payload["whiten"]),
        )
        pca.components_ = np.asarray(
            cast(list[list[float]], pca_payload["components"]),
            dtype=float,
        )
        pca.mean_ = np.asarray(
            cast(list[float], pca_payload["mean"]),
            dtype=float,
        )
        pca.explained_variance_ = np.asarray(
            cast(list[float], pca_payload["explained_variance"]),
            dtype=float,
        )
        pca.explained_variance_ratio_ = np.asarray(
            cast(list[float], pca_payload["explained_variance_ratio"]),
            dtype=float,
        )
        pca.singular_values_ = np.asarray(
            cast(list[float], pca_payload["singular_values"]),
            dtype=float,
        )
        pca.n_features_in_ = int(pca_payload["n_features_in"])
        pca.n_samples_ = int(pca_payload["n_samples"])
        pca.noise_variance_ = float(pca_payload["noise_variance"])
        pca.n_components_ = pca.components_.shape[0]

        return cls(
            columns=columns,
            n_component=n_component,
            impute_strategy=impute_strategy,
            impute_values=impute_values,
            outlier_strategy=outlier_strategy,
            iqr_multiplier=iqr_multiplier,
            outlier_lower_bounds=outlier_lower_bounds,
            outlier_upper_bounds=outlier_upper_bounds,
            winsor_lower_bounds=winsor_lower_bounds,
            winsor_upper_bounds=winsor_upper_bounds,
            scaling_model=scaling_model,
            pca=pca,
            pca_column_names=tuple(
                cast(list[str], payload["pca_column_names"])
            ),
        )

    @classmethod
    def from_transform_json(cls, payload_json: str) -> PcaModel:
        """Restore a model from serialized transformation state.

        Parameters
        ----------
        payload_json : str
            JSON produced by :meth:`to_transform_json`.

        Returns
        -------
        PcaModel
            Reconstructed model ready for transformation.
        """
        payload = cast(dict[str, object], json.loads(payload_json))
        return cls.from_transform_payload(payload)


def _validate_pca_args(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None,
    max_n_component: int | None,
    impute_strategy: Literal["drop", "median"],
    outlier_strategy: OutlierStrategy,
    iqr_multiplier: float,
    scaling_strategy: ScalingStrategy,
) -> None:
    if not columns:
        raise ValueError("columns must not be empty")
    if n_component is not None and n_component <= 0:
        raise ValueError("n_component must be greater than 0")
    if max_n_component is not None:
        if max_n_component <= 0:
            raise ValueError("max_n_component must be greater than 0")
        if n_component is not None and n_component > max_n_component:
            raise ValueError("n_component must be less than or equal to max_n_component")
    if impute_strategy not in {"drop", "median"}:
        raise ValueError("impute_strategy must be 'drop' or 'median'")
    if outlier_strategy not in {None, "winsorize", "drop"}:
        raise ValueError("outlier_strategy must be None, 'winsorize', or 'drop'")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than 0")
    if scaling_strategy not in {"none", "z-score", "minmax", "robust"}:
        raise ValueError(
            "scaling_strategy must be 'none', 'z-score', 'minmax', or 'robust'"
        )

    schema_names = set(df.collect_schema().names())
    missing_cols = [col for col in columns if col not in schema_names]
    if missing_cols:
        raise ValueError(f"columns not found in df: {missing_cols}")


def _prepare_input_dataframe(
    df: pl.LazyFrame,
    columns: list[str],
    impute_strategy: Literal["drop", "median"],
    impute_values: dict[str, float] | None = None,
) -> tuple[pl.LazyFrame, dict[str, float]]:
    if impute_strategy == "drop":
        missing_exprs = [
            pl.col(col).is_null() | pl.col(col).is_nan()
            for col in columns
        ]
        return df.filter(~pl.any_horizontal(missing_exprs)), {}

    resolved_impute_values = impute_values
    if resolved_impute_values is None:
        medians_df = cast(
            pl.DataFrame,
            df.select(
                [
                    pl.col(col).median().alias(col)
                    for col in columns
                ]
            ).collect(),
        )
        medians = medians_df.row(0, named=True)
        resolved_impute_values = {
            col: float(medians[col]) if medians[col] is not None else 0.0
            for col in columns
        }

    return (
        df.with_columns(
            [
                pl.col(col)
                .fill_null(resolved_impute_values[col])
                .fill_nan(resolved_impute_values[col])
                .alias(col)
                for col in columns
            ]
        ),
        resolved_impute_values,
    )


def fit_pca(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None = None,
    max_n_component: int | None = 100,
    impute_strategy: Literal["drop", "median"] = "drop",
    outlier_strategy: OutlierStrategy = None,
    iqr_multiplier: float = 1.5,
    scaling_strategy: ScalingStrategy = "robust",
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
    impute_strategy : {"drop", "median"}, default "drop"
        Missing-value handling strategy.
    outlier_strategy : OutlierStrategy, default None
        Optional outlier handling performed before scaling.
    iqr_multiplier : float, default 1.5
        Positive multiplier used to calculate IQR outlier bounds.
    scaling_strategy : ScalingStrategy, default "robust"
        Scaling applied before PCA fitting.

    Returns
    -------
    PcaModel
        Fitted preprocessing and PCA state.

    Raises
    ------
    ValueError
        If arguments, columns, or the prepared data are invalid.
    """
    _validate_pca_args(
        df,
        columns,
        n_component,
        max_n_component,
        impute_strategy,
        outlier_strategy,
        iqr_multiplier,
        scaling_strategy,
    )

    prepared_df, impute_values = _prepare_input_dataframe(
        df,
        columns,
        impute_strategy,
    )
    (
        prepared_df,
        outlier_lower_bounds,
        outlier_upper_bounds,
        winsor_lower_bounds,
        winsor_upper_bounds,
    ) = _prepare_outlier_dataframe(
        prepared_df,
        columns,
        outlier_strategy,
        iqr_multiplier,
    )

    scaling_model = fit_scaler(
        prepared_df,
        columns,
        scaling_strategy,
    )
    scaled_df = apply_scaler(
        prepared_df,
        columns,
        scaling_model,
    )

    standardized_df = cast(
        pl.DataFrame,
        scaled_df.select(columns).collect(),
    )
    if standardized_df.height == 0:
        raise ValueError("no rows remain after missing-value handling")

    if n_component is not None and n_component > standardized_df.height:
        raise ValueError(
            "n_component must be less than or equal to the number of samples"
        )

    upper_bound = (
        len(columns)
        if max_n_component is None
        else min(len(columns), max_n_component)
    )
    fitted_n_component = (
        min(upper_bound, standardized_df.height)
        if n_component is None
        else min(n_component, upper_bound)
    )

    pca = PCA(n_components=fitted_n_component)
    pca.fit(standardized_df.to_numpy())

    return PcaModel(
        columns=tuple(columns),
        n_component=fitted_n_component,
        impute_strategy=impute_strategy,
        impute_values=impute_values,
        outlier_strategy=outlier_strategy,
        iqr_multiplier=iqr_multiplier,
        outlier_lower_bounds=outlier_lower_bounds,
        outlier_upper_bounds=outlier_upper_bounds,
        winsor_lower_bounds=winsor_lower_bounds,
        winsor_upper_bounds=winsor_upper_bounds,
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

    _validate_pca_args(
        df,
        columns,
        pca_model.n_component,
        pca_model.n_component,
        pca_model.impute_strategy,
        pca_model.outlier_strategy,
        pca_model.iqr_multiplier,
        pca_model.scaling_model.strategy,
    )

    # Missing-value handling.
    prepared_df, _ = _prepare_input_dataframe(
        df,
        columns,
        pca_model.impute_strategy,
        pca_model.impute_values,
    )

    # Outlier handling
    prepared_df, _, _, _, _ = _prepare_outlier_dataframe(
        prepared_df,
        columns,
        pca_model.outlier_strategy,
        pca_model.iqr_multiplier,
        pca_model.outlier_lower_bounds,
        pca_model.outlier_upper_bounds,
        pca_model.winsor_lower_bounds,
        pca_model.winsor_upper_bounds,
    )

    # Feature scaling.
    scaled_df = apply_scaler(
        prepared_df,
        columns,
        pca_model.scaling_model,
    )

    standardized_df = cast(
        pl.DataFrame,
        scaled_df.select(columns).collect(),
    )
    if standardized_df.height == 0:
        raise ValueError("no rows remain after missing-value handling")

    scores = pca_model.pca.transform(standardized_df.to_numpy())
    pca_df = pl.DataFrame(
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
        scaled_df.with_row_index("__row_id")
        .join(pca_df.lazy(), on="__row_id", how="left")
        .drop("__row_id")
    )


def fit_and_transform_pca(
    df: pl.LazyFrame,
    columns: list[str],
    n_component: int | None = None,
    max_n_component: int | None = 100,
    impute_strategy: Literal["drop", "median"] = "drop",
    outlier_strategy: OutlierStrategy = None,
    iqr_multiplier: float = 1.5,
    scaling_strategy: ScalingStrategy = "robust",
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
    impute_strategy : {"drop", "median"}, default "drop"
        Missing-value handling strategy.
    outlier_strategy : OutlierStrategy, default None
        Optional outlier handling performed before scaling.
    iqr_multiplier : float, default 1.5
        Positive multiplier used to calculate IQR outlier bounds.
    scaling_strategy : ScalingStrategy, default "robust"
        Scaling applied before PCA fitting.

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
        outlier_strategy=outlier_strategy,
        iqr_multiplier=iqr_multiplier,
        scaling_strategy=scaling_strategy,
    )
    return transform_pca(df, pca_model)
