"""Reconstruction of features from PCA scores.

Scores are mapped back through the leading components to the preprocessed
(scaled) feature space, and the fitted scaling is then undone. Missing-value
imputation and outlier clipping cannot be undone, so the result approximates
the imputed and clipped data rather than the raw input. The contribution of a
single component is mapped back with the scale only, without the center.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from ..scaling import ScalingModel
from .analysis import resolve_used_components


def _validate_pca_state(pca: PCA, n_features: int) -> None:
    """Check that the PCA arrays used for reconstruction agree in shape.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    n_features : int
        Number of feature columns the model was fitted on.

    Raises
    ------
    ValueError
        If the component matrix, mean, or variances disagree in shape with
        each other or with ``n_features``.
    """
    component_matrix = np.asarray(pca.components_)
    component_count = component_matrix.shape[0]
    if np.asarray(pca.explained_variance_ratio_).shape[0] != component_count:
        raise ValueError(
            "inconsistent PCA state: explained_variance_ratio and components"
        )
    if np.asarray(pca.explained_variance_).shape[0] != component_count:
        raise ValueError("inconsistent PCA state: explained_variance and components")
    if component_matrix.shape[1] != n_features:
        raise ValueError("inconsistent PCA state: components and feature columns")
    if np.asarray(pca.mean_).shape[0] != n_features:
        raise ValueError("inconsistent PCA state: mean and feature columns")


def reconstruct_standardized(
    scores: np.ndarray,
    pca: PCA,
    used_components: int,
) -> np.ndarray:
    """Reconstruct preprocessed features from the leading PCA scores.

    Computes ``mean + sum_{j<k} t_j w_j``. For a whitened PCA the scores are
    first multiplied back by ``sqrt(explained_variance)``, matching
    ``PCA.inverse_transform``.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores of shape ``(n_rows, n)`` with ``n >= used_components``;
        columns beyond ``used_components`` are ignored. NaN in a used score
        makes the whole reconstructed row NaN.
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    used_components : int
        Number of leading components ``k`` to reconstruct from.

    Returns
    -------
    np.ndarray
        Reconstructed values of shape ``(n_rows, n_features)`` in the
        preprocessed feature space.

    Raises
    ------
    ValueError
        If ``used_components`` is outside ``1..`` the available components.
    """
    component_matrix = np.asarray(pca.components_, dtype=float)
    score_matrix = np.asarray(scores, dtype=float)
    if (
        used_components < 1
        or used_components > component_matrix.shape[0]
        or used_components > score_matrix.shape[1]
    ):
        raise ValueError("used_components must be between 1 and the component count")

    used_scores = _unwhitened_scores(score_matrix, pca, slice(0, used_components))
    return used_scores @ component_matrix[:used_components] + np.asarray(
        pca.mean_, dtype=float
    )


def _unwhitened_scores(
    score_matrix: np.ndarray, pca: PCA, components: slice
) -> np.ndarray:
    """Return score columns in the unwhitened component scale.

    Parameters
    ----------
    score_matrix : np.ndarray
        PCA scores of shape ``(n_rows, n)``.
    pca : PCA
        Fitted scikit-learn PCA estimator that produced the scores.
    components : slice
        Zero-based component columns to return.

    Returns
    -------
    np.ndarray
        The selected score columns, multiplied by
        ``sqrt(explained_variance)`` for a whitened PCA as in
        ``PCA.inverse_transform``.
    """
    used_scores = score_matrix[:, components]
    if pca.whiten:
        used_scores = used_scores * np.sqrt(
            np.asarray(pca.explained_variance_, dtype=float)[components]
        )
    return used_scores


def component_contribution(
    scores: np.ndarray,
    pca: PCA,
    scaling_model: ScalingModel,
    columns: tuple[str, ...],
    component: int,
) -> np.ndarray:
    """Return the contribution of one component in the original scale.

    Computes ``t_k w_k`` in the preprocessed space and multiplies it by the
    fitted scale without adding the center, so the contributions of all
    components plus the mean in the original scale add up to the full
    reconstruction.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores of shape ``(n_rows, n)`` with ``n >= component``. NaN in
        the component's score makes the row NaN.
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    scaling_model : ScalingModel
        Fitted scaling state whose scales are applied. ``strategy ==
        "none"`` applies no scale.
    columns : tuple[str, ...]
        Feature columns in the order they were fitted.
    component : int
        One-based component number ``k``.

    Returns
    -------
    np.ndarray
        Contribution of shape ``(n_rows, len(columns))``.

    Raises
    ------
    ValueError
        If ``component`` is outside ``1..`` the available components, or
        the PCA or scaling state is inconsistent.
    """
    _validate_pca_state(pca, len(columns))
    component_matrix = np.asarray(pca.components_, dtype=float)
    score_matrix = np.asarray(scores, dtype=float)
    if (
        component < 1
        or component > component_matrix.shape[0]
        or component > score_matrix.shape[1]
    ):
        raise ValueError("component must be between 1 and the component count")
    used = slice(component - 1, component)
    term = _unwhitened_scores(score_matrix, pca, used) @ component_matrix[used]
    return unscale(term, scaling_model, columns, center=False)


def unscale(
    values: np.ndarray,
    scaling_model: ScalingModel,
    columns: tuple[str, ...],
    *,
    center: bool = True,
) -> np.ndarray:
    """Undo the fitted feature scaling, ``value * scale + center``.

    Parameters
    ----------
    values : np.ndarray
        Scaled values of shape ``(n_rows, len(columns))``.
    scaling_model : ScalingModel
        Fitted scaling state. ``strategy == "none"`` returns ``values``
        unchanged.
    columns : tuple[str, ...]
        Feature column of each value column, in order.
    center : bool, default True
        Whether to add the center. ``False`` only multiplies by the scale,
        which maps a difference of scaled values to the original scale.

    Returns
    -------
    np.ndarray
        Values in the original feature scale.

    Raises
    ------
    ValueError
        If the scaling model lacks the center or scale of a column.
    """
    if scaling_model.strategy == "none":
        return values
    missing_columns = [
        column
        for column in columns
        if column not in scaling_model.centers or column not in scaling_model.scales
    ]
    if missing_columns:
        raise ValueError(
            f"scaling model has no center or scale for columns: {missing_columns}"
        )
    scales = np.array([scaling_model.scales[column] for column in columns], dtype=float)
    if not center:
        return values * scales
    centers = np.array(
        [scaling_model.centers[column] for column in columns], dtype=float
    )
    return values * scales + centers


def reconstruct_features(
    scores: pl.DataFrame,
    pca: PCA,
    scaling_model: ScalingModel,
    columns: tuple[str, ...],
    pca_column_names: tuple[str, ...],
    cumulative_explained_variance: float | None = None,
) -> pl.DataFrame:
    """Reconstruct features in the original scale from a score frame.

    Parameters
    ----------
    scores : pl.DataFrame
        Frame holding at least the first ``k`` score columns of
        ``pca_column_names``.
    pca : PCA
        Fitted scikit-learn PCA estimator.
    scaling_model : ScalingModel
        Fitted scaling state to undo.
    columns : tuple[str, ...]
        Feature columns in the order they were fitted.
    pca_column_names : tuple[str, ...]
        Score-column name of each fitted component.
    cumulative_explained_variance : float | int | None, default None
        Component selector for the leading components; see
        ``resolve_used_components``. ``None`` uses every fitted component.

    Returns
    -------
    pl.DataFrame
        The non-score columns of ``scores`` followed by the reconstructed
        feature columns in ``columns`` order, with the input row order.
        Input columns named like a feature column are replaced by the
        reconstructed values. Rows with a null or NaN used score are NaN.

    Raises
    ------
    TypeError
        If ``scores`` is not a ``pl.DataFrame``.
    ValueError
        If the selector is out of range, a used score column is missing,
        or the PCA or scaling state is inconsistent.
    """
    if not isinstance(scores, pl.DataFrame):
        raise TypeError("scores must be a polars DataFrame")
    _validate_pca_state(pca, len(columns))
    component_count = np.asarray(pca.components_).shape[0]
    if len(pca_column_names) != component_count:
        raise ValueError("inconsistent PCA state: pca_column_names and components")

    used_components = resolve_used_components(
        np.asarray(pca.explained_variance_ratio_, dtype=float),
        component_count,
        cumulative_explained_variance,
    )
    used_score_columns = list(pca_column_names[:used_components])
    missing_columns = [
        column for column in used_score_columns if column not in scores.columns
    ]
    if missing_columns:
        raise ValueError(f"score columns not found in scores: {missing_columns}")

    score_matrix = (
        scores.select(pl.col(used_score_columns).cast(pl.Float64))
        .to_numpy()
        .astype(float)
    )
    standardized = reconstruct_standardized(score_matrix, pca, used_components)
    reconstructed = pl.from_numpy(
        unscale(standardized, scaling_model, columns),
        schema=list(columns),
        orient="row",
    )

    excluded_columns = set(pca_column_names) | set(columns)
    passthrough = scores.select(
        [column for column in scores.columns if column not in excluded_columns]
    )
    if passthrough.width == 0:
        return reconstructed
    return passthrough.hstack(reconstructed.get_columns())
