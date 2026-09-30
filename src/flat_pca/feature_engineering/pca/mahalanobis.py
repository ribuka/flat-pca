"""Mahalanobis distance and control limit in the PCA score space.

In the score space of a fitted PCA the covariance matrix is diagonal, with
each component's variance ``explained_variance_`` on the diagonal. The
squared Mahalanobis distance therefore reduces to the Hotelling T² form
``sum_j t_j**2 / lambda_j`` and stays well defined even when the original
feature space has far more dimensions than samples.

The component selector defaults to ``0.9`` (cumulative explained variance)
rather than every fitted component, because ``flatten_pca`` fits as many
components as samples by default: the trailing components then have
near-zero variance, which makes the distance diverge, and with
``k = n - 1`` every training sample has the same distance
``(n - 1)**2 / n``, so anomalies cannot be told apart.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import f as f_distribution
from sklearn.decomposition import PCA

from .analysis import resolve_used_components

# Component variances at or below this fraction of the largest used variance
# are treated as zero, because dividing by them makes the distance diverge.
RELATIVE_VARIANCE_TOLERANCE = 1e-12


def _validate_alpha(alpha: float) -> None:
    """Check that a significance level lies strictly between 0 and 1.

    Parameters
    ----------
    alpha : float
        Significance level of the control limit.

    Raises
    ------
    ValueError
        If ``alpha`` is not a real number with ``0 < alpha < 1``.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        # Keep ValueError as the single contract for rejected settings.
        raise ValueError("alpha must satisfy 0 < alpha < 1")  # noqa: TRY004
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1")


@dataclass(frozen=True)
class MahalanobisConfig:
    """Settings for appending Mahalanobis distance columns to PCA scores.

    Attributes
    ----------
    cumulative_explained_variance : float | int | None, default 0.9
        Component selector for the leading components used in the distance,
        following the same rule as ``resolve_used_components``. The default
        differs from ``get_feature_contribution_ranking`` (``None``) on
        purpose: using every component fitted by ``flatten_pca`` includes
        near-zero-variance components and makes the training distances
        indistinguishable.
    alpha : float, default 0.01
        Significance level of the upper control limit, ``0 < alpha < 1``.
    distance_column : str, default "mahalanobis_sq"
        Column holding the squared Mahalanobis distance.
    ucl_column : str, default "mahalanobis_ucl"
        Column holding the upper control limit (same value on every row).
    exceeds_ucl_column : str, default "mahalanobis_exceeds_ucl"
        Boolean column holding ``distance > UCL``.

    Raises
    ------
    ValueError
        If ``alpha`` is out of range, or a column name is empty or repeated.
    """

    cumulative_explained_variance: float | int | None = 0.9
    alpha: float = 0.01
    distance_column: str = "mahalanobis_sq"
    ucl_column: str = "mahalanobis_ucl"
    exceeds_ucl_column: str = "mahalanobis_exceeds_ucl"

    def __post_init__(self) -> None:
        """Validate the significance level and output column names.

        Raises
        ------
        ValueError
            If ``alpha`` is out of range, or a column name is empty or
            repeated.
        """
        _validate_alpha(self.alpha)
        column_names = self.column_names
        for column_name in column_names:
            if not isinstance(column_name, str) or not column_name:
                raise ValueError("Mahalanobis column names must be non-empty strings")
        if len(set(column_names)) != len(column_names):
            raise ValueError("Mahalanobis column names must be distinct")

    @property
    def column_names(self) -> tuple[str, str, str]:
        """Return the output column names in their appended order.

        Returns
        -------
        tuple[str, str, str]
            Distance, UCL, and exceeds-UCL column names.
        """
        return (self.distance_column, self.ucl_column, self.exceeds_ucl_column)


def _validate_variances(explained_variance: np.ndarray) -> None:
    """Reject component variances that would make the distance diverge.

    Parameters
    ----------
    explained_variance : np.ndarray
        Variances of the components used in the distance.

    Raises
    ------
    ValueError
        If any variance is non-finite or effectively zero relative to the
        largest one.
    """
    if not np.all(np.isfinite(explained_variance)):
        raise ValueError("explained variance of the used components must be finite")
    largest = float(np.max(explained_variance))
    if largest <= 0 or np.any(
        explained_variance <= largest * RELATIVE_VARIANCE_TOLERANCE
    ):
        raise ValueError(
            "a used component has near-zero variance; "
            "reduce cumulative_explained_variance"
        )


def mahalanobis_distance_sq(
    scores: np.ndarray,
    explained_variance: np.ndarray,
    used_components: int,
) -> np.ndarray:
    """Return the squared Mahalanobis distance of each row in score space.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores of shape ``(n_rows, n_components)``.
    explained_variance : np.ndarray
        Variance of each component, in the same order as the score columns.
    used_components : int
        Number of leading components ``k`` used in the distance.

    Returns
    -------
    np.ndarray
        ``sum_{j<k} scores[:, j]**2 / explained_variance[j]`` per row.

    Raises
    ------
    ValueError
        If ``used_components`` exceeds the available components, or a used
        variance is non-finite or effectively zero.
    """
    variances = np.asarray(explained_variance, dtype=float)
    if (
        used_components < 1
        or used_components > variances.shape[0]
        or used_components > scores.shape[1]
    ):
        raise ValueError("used_components must be between 1 and the component count")
    used_variances = variances[:used_components]
    _validate_variances(used_variances)
    used_scores = np.asarray(scores, dtype=float)[:, :used_components]
    return np.sum(used_scores**2 / used_variances, axis=1)


def mahalanobis_ucl(n_samples: int, used_components: int, alpha: float) -> float:
    """Return the F-distribution upper control limit for new observations.

    ``UCL = k (n + 1)(n - 1) / (n (n - k)) * F_{1-alpha}(k, n - k)``.

    Parameters
    ----------
    n_samples : int
        Number of samples ``n`` the PCA was fitted on.
    used_components : int
        Number of leading components ``k`` used in the distance.
    alpha : float
        Significance level, ``0 < alpha < 1``.

    Returns
    -------
    float
        Upper control limit of the squared Mahalanobis distance.

    Raises
    ------
    ValueError
        If ``alpha`` is out of range, ``used_components`` is below 1, or
        ``used_components >= n_samples`` leaves no F-distribution degrees of
        freedom.
    """
    _validate_alpha(alpha)
    if used_components < 1:
        raise ValueError("used_components must be at least 1")
    denominator_dof = n_samples - used_components
    if denominator_dof <= 0:
        raise ValueError(
            "used component count must be less than the fitted sample count "
            f"(used_components={used_components}, n_samples={n_samples})"
        )
    quantile = float(
        f_distribution.ppf(1.0 - float(alpha), used_components, denominator_dof)
    )
    scale = (
        used_components
        * (n_samples + 1)
        * (n_samples - 1)
        / (n_samples * denominator_dof)
    )
    return scale * quantile


def resolve_mahalanobis_components(
    pca: PCA,
    cumulative_explained_variance: float | None,
) -> int:
    """Resolve the number of components a selector picks from a fitted PCA.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    cumulative_explained_variance : float | int | None
        Component selector; see ``resolve_used_components``.

    Returns
    -------
    int
        Number of leading components to use.
    """
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    return resolve_used_components(
        explained_variance_ratio,
        explained_variance_ratio.shape[0],
        cumulative_explained_variance,
    )


def mahalanobis_threshold(
    pca: PCA,
    alpha: float,
    cumulative_explained_variance: float | None,
) -> float:
    """Return the upper control limit for a fitted PCA.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    alpha : float
        Significance level, ``0 < alpha < 1``.
    cumulative_explained_variance : float | int | None
        Component selector; see ``resolve_used_components``.

    Returns
    -------
    float
        Upper control limit of the squared Mahalanobis distance.
    """
    used_components = resolve_mahalanobis_components(
        pca, cumulative_explained_variance
    )
    return mahalanobis_ucl(int(pca.n_samples_), used_components, alpha)


def pca_mahalanobis_distance_sq(
    pca: PCA,
    scores: np.ndarray,
    used_components: int,
) -> np.ndarray:
    """Return squared Mahalanobis distances of scores from a fitted PCA.

    A whitened PCA already divides each score by ``sqrt(lambda_j)``, so its
    scores are compared against unit variances instead.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    scores : np.ndarray
        PCA scores of shape ``(n_rows, n_components)``.
    used_components : int
        Number of leading components used in the distance.

    Returns
    -------
    np.ndarray
        Squared Mahalanobis distance per row.

    Raises
    ------
    ValueError
        If a used component has non-finite or near-zero variance.
    """
    explained_variance = np.asarray(pca.explained_variance_, dtype=float)
    if not pca.whiten:
        return mahalanobis_distance_sq(scores, explained_variance, used_components)
    _validate_variances(explained_variance[:used_components])
    return mahalanobis_distance_sq(
        scores, np.ones_like(explained_variance), used_components
    )


def validate_mahalanobis_request(
    pca: PCA,
    config: MahalanobisConfig,
    existing_columns: Collection[str],
) -> None:
    """Check that Mahalanobis columns can be appended before transforming.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    config : MahalanobisConfig
        Requested distance settings.
    existing_columns : Collection[str]
        Columns already present in the output (input and score columns).

    Raises
    ------
    ValueError
        If a Mahalanobis column collides with an existing column, the
        component selector is invalid, too many components are selected for
        the fitted sample count, or a used component has near-zero variance.
    """
    colliding = [name for name in config.column_names if name in existing_columns]
    if colliding:
        raise ValueError(f"Mahalanobis columns collide with existing columns: {colliding}")
    used_components = resolve_mahalanobis_components(
        pca, config.cumulative_explained_variance
    )
    mahalanobis_ucl(int(pca.n_samples_), used_components, config.alpha)
    _validate_variances(
        np.asarray(pca.explained_variance_, dtype=float)[:used_components]
    )


def mahalanobis_score_columns(
    pca: PCA,
    scores: np.ndarray,
    config: MahalanobisConfig,
) -> dict[str, np.ndarray]:
    """Return the distance, UCL, and exceeds-UCL columns for PCA scores.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    scores : np.ndarray
        PCA scores of shape ``(n_rows, n_components)``.
    config : MahalanobisConfig
        Distance settings, including the output column names.

    Returns
    -------
    dict[str, np.ndarray]
        Column name to values, in the order distance, UCL, exceeds-UCL.
    """
    used_components = resolve_mahalanobis_components(
        pca, config.cumulative_explained_variance
    )
    distance = pca_mahalanobis_distance_sq(pca, scores, used_components)
    ucl = mahalanobis_ucl(int(pca.n_samples_), used_components, config.alpha)
    return {
        config.distance_column: distance,
        config.ucl_column: np.full(distance.shape[0], ucl),
        config.exceeds_ucl_column: distance > ucl,
    }
