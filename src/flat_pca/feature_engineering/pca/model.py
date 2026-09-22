"""Fitted PCA pipeline state.

The model is a thin container: component interpretation lives in
``analysis`` and payload conversion in ``serialization``, and the methods
here only forward to them so the dataclass stays readable as a description
of the fitted state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

import polars as pl
from sklearn.decomposition import PCA

from ..outlier import OutlierStrategy
from ..scaling import ScalingModel
from .analysis import component_coefficients, feature_contribution_ranking
from .serialization import build_transform_payload, parse_transform_payload


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
        return component_coefficients(
            self.pca,
            self.columns,
            self.n_component,
            component,
        )

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
        return feature_contribution_ranking(
            self.pca,
            self.columns,
            cumulative_explained_variance,
            include_component_breakdown,
            component_prefix,
        )

    def to_transform_payload(self) -> dict[str, object]:
        """Return parameters required to reproduce transformation.

        Returns
        -------
        dict[str, object]
            JSON-serializable preprocessing and PCA state.
        """
        return build_transform_payload(self)

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
        return cls(**parse_transform_payload(payload))  # type: ignore[arg-type]

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
