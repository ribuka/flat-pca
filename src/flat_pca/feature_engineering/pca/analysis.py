"""Interpretation of fitted PCA components.

These functions take the fitted scikit-learn estimator and its feature
column names rather than a ``PcaModel``, so the analysis logic stays
independent of the model container and can be tested on its own.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import polars as pl
from sklearn.decomposition import PCA


def component_coefficients(
    pca: PCA,
    columns: tuple[str, ...],
    n_component: int,
    component: int,
) -> pl.DataFrame:
    """Return coefficients for one component ordered by absolute value.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    columns : tuple[str, ...]
        Feature columns in the order they were fitted.
    n_component : int
        Number of fitted principal components.
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
    if component > n_component:
        raise ValueError("component must be less than or equal to n_component")

    component_index = component - 1
    coefficients = pca.components_[component_index]
    rows = [
        {
            "column": column,
            "coefficient": round(float(coefficients[index]), 4),
            "abs_coefficient": round(float(abs(coefficients[index])), 4),
        }
        for index, column in enumerate(columns)
    ]
    return pl.DataFrame(rows).sort("abs_coefficient", descending=True)


def resolve_used_components(
    explained_variance_ratio: np.ndarray,
    component_count: int,
    cumulative_explained_variance: float | None,
) -> int:
    """Resolve how many leading components a component selector selects.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained-variance ratio of each fitted component.
    component_count : int
        Number of fitted components.
    cumulative_explained_variance : float | int | None
        Component selector. ``None`` selects every fitted component, an
        ``int`` (>=1) caps the component count, and a ``float`` in
        ``(0, 1]`` is a cumulative explained-variance threshold.

    Returns
    -------
    int
        Number of leading components to use, never exceeding
        ``component_count``.

    Raises
    ------
    ValueError
        If the selector is out of range or unsupported in type.
    """
    if cumulative_explained_variance is None:
        return component_count
    if isinstance(cumulative_explained_variance, bool):
        # The documented contract for every rejected selector, including a
        # wrong type, is ValueError; keep it rather than raising TypeError.
        raise ValueError(  # noqa: TRY004
            "cumulative_explained_variance must be None, int>=1, or float in (0, 1]"
        )
    if isinstance(cumulative_explained_variance, Integral):
        component_cap = int(cumulative_explained_variance)
        if component_cap < 1:
            raise ValueError("cumulative_explained_variance as int must be >= 1")
        return min(component_cap, component_count)
    if isinstance(cumulative_explained_variance, Real):
        variance_threshold = float(cumulative_explained_variance)
        if variance_threshold <= 0 or variance_threshold > 1:
            raise ValueError(
                "cumulative_explained_variance as float must satisfy 0 < value <= 1"
            )
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        used_components = int(
            np.searchsorted(cumulative_ratio, variance_threshold, side="left") + 1
        )
        return min(used_components, component_count)
    raise ValueError(
        "cumulative_explained_variance must be None, int>=1, or float in (0, 1]"
    )


def feature_contribution_ranking(
    pca: PCA,
    columns: tuple[str, ...],
    cumulative_explained_variance: float | None = None,
    include_component_breakdown: bool = False,
    component_prefix: str = "contribution_pc",
) -> pl.DataFrame:
    """Return ranked feature contributions decomposed from explained variance.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    columns : tuple[str, ...]
        Feature columns in the order they were fitted.
    cumulative_explained_variance : float | int | None, default None
        Component selector for leading components. See
        ``resolve_used_components``.
    include_component_breakdown : bool, default False
        If ``True``, append per-component contribution columns such as
        ``contribution_pc1``, ``contribution_pc2``, and so on.
    component_prefix : str, default "contribution_pc"
        Prefix used when naming per-component contribution columns.

    Returns
    -------
    pl.DataFrame
        Ranking table sorted by ``contribution`` in descending order with
        columns ``rank``, ``feature``, ``contribution``, ``used_components``,
        and ``used_cumulative_explained_variance``. When
        ``include_component_breakdown`` is ``True``, per-component
        contribution columns are included as well.

    Raises
    ------
    ValueError
        If ``cumulative_explained_variance`` is out of range, unsupported in
        type, or if the PCA state is inconsistent.
    """
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    component_matrix = np.asarray(pca.components_, dtype=float)
    component_count = component_matrix.shape[0]

    if explained_variance_ratio.shape[0] != component_count:
        raise ValueError(
            "inconsistent PCA state: explained_variance_ratio and components"
        )
    if component_matrix.shape[1] != len(columns):
        raise ValueError("inconsistent PCA state: components and feature columns")

    used_components = resolve_used_components(
        explained_variance_ratio,
        component_count,
        cumulative_explained_variance,
    )

    used_component_matrix = component_matrix[:used_components]
    used_explained_variance_ratio = explained_variance_ratio[:used_components]
    component_contribution_matrix = (
        (used_component_matrix**2) * used_explained_variance_ratio[:, np.newaxis]
    )
    contribution_by_feature = component_contribution_matrix.sum(axis=0)
    used_cumulative_explained_variance = float(used_explained_variance_ratio.sum())

    frame_data: dict[str, object] = {
        "feature": list(columns),
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

    return (
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
