"""Fitted PCA pipeline state.

The model is a thin container: component interpretation lives in
``analysis``, score-space distances in ``mahalanobis``, residual Q
statistics in ``spe``, feature reconstruction in ``reconstruct``, and payload conversion in
``serialization``, and the methods
here only forward to them so the dataclass stays readable as a description
of the fitted state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import polars as pl
from sklearn.decomposition import PCA

from ..outlier import OutlierModel
from ..scaling import ScalingModel
from .analysis import (
    component_coefficients,
    explained_variance_table,
    feature_contribution_ranking,
)
from .impute import ImputeModel
from .mahalanobis import mahalanobis_threshold
from .reconstruct import reconstruct_features
from .serialization import build_transform_payload, parse_transform_payload
from .spe import spe_threshold


@dataclass(frozen=True)
class PcaModel:
    """Store a fitted PCA pipeline and its preprocessing state.

    Attributes
    ----------
    columns : tuple[str, ...]
        Feature columns expected by the model.
    n_component : int
        Number of fitted principal components.
    impute_model : ImputeModel
        Fitted missing-value handling state, applied first.
    outlier_model : OutlierModel
        Fitted outlier-handling state, applied after imputation.
    scaling_model : ScalingModel
        Fitted feature-scaling state, applied last before PCA.
    pca : PCA
        Fitted scikit-learn PCA estimator.
    pca_column_names : tuple[str, ...]
        Output score-column names.
    """

    columns: tuple[str, ...]
    n_component: int
    impute_model: ImputeModel
    outlier_model: OutlierModel
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

    def get_explained_variance_table(self) -> pl.DataFrame:
        """Return the explained variance of every fitted component.

        The cumulative values come from the same computation that
        ``cumulative_explained_variance`` selectors use, so the component
        count chosen for a threshold matches the first row whose cumulative
        value reaches it.

        Returns
        -------
        pl.DataFrame
            One row per component in ascending ``component`` order, with
            columns ``component`` (one-based), ``pca_column``,
            ``explained_variance``, ``explained_variance_ratio``, and
            ``cumulative_explained_variance``.

        Raises
        ------
        ValueError
            If the PCA state is inconsistent.
        """
        return explained_variance_table(self.pca, self.pca_column_names)

    def reconstruct(
        self,
        scores: pl.DataFrame,
        cumulative_explained_variance: float | None = None,
    ) -> pl.DataFrame:
        """Reconstruct features in the original scale from PCA scores.

        Computes ``mean + sum_{j<=k} t_j w_j`` in the preprocessed space and
        undoes the fitted scaling. Imputation and outlier clipping cannot be
        undone, so the result approximates the imputed and clipped data.

        Parameters
        ----------
        scores : pl.DataFrame
            Frame holding at least the first ``k`` score columns of
            ``pca_column_names``; later score columns are ignored.
        cumulative_explained_variance : float | int | None, default None
            Component selector for leading components, following the same
            rule as :meth:`get_feature_contribution_ranking`. ``None`` uses
            every fitted component.

        Returns
        -------
        pl.DataFrame
            The non-score columns of ``scores`` followed by the reconstructed
            feature columns in ``columns`` order, with the input row order.
            Input columns named like a feature column are replaced. Rows
            with a null or NaN used score are NaN.

        Raises
        ------
        TypeError
            If ``scores`` is not a ``pl.DataFrame``.
        ValueError
            If the selector is out of range, a used score column is missing,
            or the PCA or scaling state is inconsistent.
        """
        return reconstruct_features(
            scores,
            self.pca,
            self.scaling_model,
            self.columns,
            self.pca_column_names,
            cumulative_explained_variance,
        )

    def get_mahalanobis_threshold(
        self,
        alpha: float = 0.01,
        cumulative_explained_variance: float | None = 0.9,
    ) -> float:
        """Return the upper control limit of the squared Mahalanobis distance.

        The limit is the F-distribution UCL for new observations,
        ``k (n + 1)(n - 1) / (n (n - k)) * F_{1-alpha}(k, n - k)``, where
        ``n`` is the fitted sample count and ``k`` the used component count.

        Parameters
        ----------
        alpha : float, default 0.01
            Significance level, ``0 < alpha < 1``.
        cumulative_explained_variance : float | int | None, default 0.9
            Component selector for leading components, following the same
            rule as :meth:`get_feature_contribution_ranking`. The default
            avoids the near-zero-variance trailing components that
            ``flatten_pca`` fits by default.

        Returns
        -------
        float
            Upper control limit, e.g. for a horizontal line on a plot.

        Raises
        ------
        ValueError
            If ``alpha`` or the component selector is out of range, or the
            used component count is not below the fitted sample count.
        """
        return mahalanobis_threshold(
            self.pca,
            alpha,
            cumulative_explained_variance,
        )

    def get_spe_threshold(
        self,
        alpha: float = 0.01,
        cumulative_explained_variance: float | None = 0.9,
    ) -> float:
        """Return the upper control limit of the Q statistic (SPE).

        The limit is the Jackson-Mudholkar approximation built from the
        variances of the components after the leading ``k``. Components
        that were not fitted are each given ``pca.noise_variance_``.

        Parameters
        ----------
        alpha : float, default 0.01
            Significance level, ``0 < alpha < 1``.
        cumulative_explained_variance : float | int | None, default 0.9
            Component selector for the leading components kept in the
            reconstruction, following the same rule as
            :meth:`get_feature_contribution_ranking`.

        Returns
        -------
        float
            Upper control limit, e.g. for a horizontal line on a plot.

        Raises
        ------
        ValueError
            If ``alpha`` or the component selector is out of range, or no
            residual component has variance.
        """
        return spe_threshold(
            self.pca,
            alpha,
            cumulative_explained_variance,
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
