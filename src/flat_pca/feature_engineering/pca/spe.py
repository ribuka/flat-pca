"""Q statistic (SPE) and control limit of the PCA residual.

Hotelling T² (``mahalanobis``) only sees variation inside the principal
subspace. A change of shape never seen during fitting, such as a new peak or
a different baseline, shows up outside that subspace instead, and the
squared prediction error ``Q = ||x - x_hat||**2`` of the reconstruction from
the leading ``k`` components measures it.

The control limit follows Jackson and Mudholkar, built from the variances of
the residual components ``k < j <= r`` with ``r = min(n_samples,
n_features)``. The variances of components that were never fitted are not
stored, so each of them is taken to be ``pca.noise_variance_``, their mean,
which keeps the payload format unchanged.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import norm as normal_distribution
from sklearn.decomposition import PCA

from .mahalanobis import (
    RELATIVE_VARIANCE_TOLERANCE,
    resolve_mahalanobis_components,
    validate_alpha,
    validate_output_column_names,
)
from .reconstruct import reconstruct_standardized


@dataclass(frozen=True)
class SpeConfig:
    """Settings for appending Q statistic (SPE) columns to PCA scores.

    Attributes
    ----------
    cumulative_explained_variance : float | int | None, default 0.9
        Component selector for the leading components ``k`` kept in the
        reconstruction, following the same rule as
        ``resolve_used_components``. The default matches
        ``MahalanobisConfig`` so that T² and Q split the same subspace.
    alpha : float, default 0.01
        Significance level of the upper control limit, ``0 < alpha < 1``.
    spe_column : str, default "spe"
        Column holding the Q statistic.
    ucl_column : str, default "spe_ucl"
        Column holding the upper control limit (same value on every row).
    exceeds_ucl_column : str, default "spe_exceeds_ucl"
        Boolean column holding ``Q > UCL``.

    Raises
    ------
    ValueError
        If ``alpha`` is out of range, or a column name is empty or repeated.
    """

    cumulative_explained_variance: float | int | None = 0.9
    alpha: float = 0.01
    spe_column: str = "spe"
    ucl_column: str = "spe_ucl"
    exceeds_ucl_column: str = "spe_exceeds_ucl"

    def __post_init__(self) -> None:
        """Validate the significance level and output column names.

        Raises
        ------
        ValueError
            If ``alpha`` is out of range, or a column name is empty or
            repeated.
        """
        validate_alpha(self.alpha)
        validate_output_column_names(self.column_names, "SPE")

    @property
    def column_names(self) -> tuple[str, str, str]:
        """Return the output column names in their appended order.

        Returns
        -------
        tuple[str, str, str]
            Q statistic, UCL, and exceeds-UCL column names.
        """
        return (self.spe_column, self.ucl_column, self.exceeds_ucl_column)


def spe(
    standardized_values: np.ndarray,
    scores: np.ndarray,
    pca: PCA,
    used_components: int,
) -> np.ndarray:
    """Return the squared reconstruction error of each row.

    ``pca.inverse_transform`` always uses every fitted component, so the
    reconstruction from the leading ``k`` scores is built with
    ``reconstruct_standardized`` instead.

    Parameters
    ----------
    standardized_values : np.ndarray
        Preprocessed feature matrix of shape ``(n_rows, n_features)`` that
        was passed to ``pca.transform``.
    scores : np.ndarray
        PCA scores of ``standardized_values``, shape ``(n_rows, n)`` with
        ``n >= used_components``.
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    used_components : int
        Number of leading components ``k`` kept in the reconstruction.

    Returns
    -------
    np.ndarray
        ``sum((x - x_hat)**2)`` per row.

    Raises
    ------
    ValueError
        If ``used_components`` is outside ``1..`` the available components.
    """
    reconstructed = reconstruct_standardized(scores, pca, used_components)
    residual = np.asarray(standardized_values, dtype=float) - reconstructed
    return np.sum(residual**2, axis=1)


def _unfitted_component_count(
    n_samples: int, n_features: int, fitted_components: int
) -> int:
    """Return how many components up to the rank bound were not fitted.

    Parameters
    ----------
    n_samples : int
        Number of samples the PCA was fitted on.
    n_features : int
        Number of features the PCA was fitted on.
    fitted_components : int
        Number of fitted components.

    Returns
    -------
    int
        ``max(min(n_samples, n_features) - fitted_components, 0)``.
    """
    return max(min(n_samples, n_features) - fitted_components, 0)


def residual_variance_moments(
    explained_variance: np.ndarray,
    noise_variance: float,
    n_samples: int,
    n_features: int,
    used_components: int,
) -> tuple[float, float, float]:
    """Return ``theta_i = sum_{j>k} lambda_j**i`` for ``i = 1, 2, 3``.

    Parameters
    ----------
    explained_variance : np.ndarray
        Variance of each fitted component, in descending order.
    noise_variance : float
        Mean variance of the components that were not fitted, used for each
        of them.
    n_samples : int
        Number of samples the PCA was fitted on.
    n_features : int
        Number of features the PCA was fitted on.
    used_components : int
        Number of leading components ``k`` kept in the reconstruction.

    Returns
    -------
    tuple[float, float, float]
        ``theta_1``, ``theta_2``, and ``theta_3``.

    Raises
    ------
    ValueError
        If ``used_components`` is outside ``1..`` the fitted components.
    """
    variances = np.asarray(explained_variance, dtype=float)
    fitted_components = variances.shape[0]
    if used_components < 1 or used_components > fitted_components:
        raise ValueError("used_components must be between 1 and the component count")
    residual_variances = variances[used_components:]
    unfitted_components = _unfitted_component_count(
        n_samples, n_features, fitted_components
    )
    return tuple(  # type: ignore[return-value]
        float(np.sum(residual_variances**power))
        + unfitted_components * float(noise_variance) ** power
        for power in (1, 2, 3)
    )


def _box_ucl(theta_1: float, theta_2: float, alpha: float) -> float:
    """Return the scaled chi-squared control limit ``g * chi2_{1-alpha}(h)``.

    Box's approximation ``Q ~ g chi2(h)`` with ``g = theta_2 / theta_1`` and
    ``h = theta_1**2 / theta_2`` is the one Jackson and Mudholkar start from.
    It stays valid where their normalizing power ``h0`` does not.

    Parameters
    ----------
    theta_1 : float
        Sum of the residual variances.
    theta_2 : float
        Sum of the squared residual variances.
    alpha : float
        Significance level, ``0 < alpha < 1``.

    Returns
    -------
    float
        Upper control limit of Q.
    """
    scale = theta_2 / theta_1
    degrees_of_freedom = theta_1**2 / theta_2
    # ``isf`` keeps a tiny ``alpha`` instead of rounding ``1 - alpha`` to 1.
    return scale * float(chi2_distribution.isf(alpha, degrees_of_freedom))


def spe_ucl(
    explained_variance: np.ndarray,
    noise_variance: float,
    n_samples: int,
    n_features: int,
    used_components: int,
    alpha: float,
) -> float:
    """Return the Jackson-Mudholkar upper control limit of Q.

    ``UCL = theta_1 [z sqrt(2 theta_2 h0**2) / theta_1 + 1
    + theta_2 h0 (h0 - 1) / theta_1**2] ** (1 / h0)`` with
    ``h0 = 1 - 2 theta_1 theta_3 / (3 theta_2**2)`` and ``z`` the standard
    normal quantile with upper-tail probability ``alpha``. When ``h0 <= 0``
    the power transform reverses the order of Q, and when the bracket is not
    positive the power is undefined, so Box's approximation
    ``theta_2 / theta_1 * chi2_{1-alpha}(theta_1**2 / theta_2)`` is used
    instead.

    Parameters
    ----------
    explained_variance : np.ndarray
        Variance of each fitted component, in descending order.
    noise_variance : float
        Mean variance of the components that were not fitted.
    n_samples : int
        Number of samples the PCA was fitted on.
    n_features : int
        Number of features the PCA was fitted on.
    used_components : int
        Number of leading components ``k`` kept in the reconstruction.
    alpha : float
        Significance level, ``0 < alpha < 1``.

    Returns
    -------
    float
        Upper control limit of Q.

    Raises
    ------
    ValueError
        If ``alpha`` is out of range, ``used_components`` is outside the
        fitted components, or no residual component has variance, e.g.
        because ``used_components`` reaches the rank bound.
    """
    validate_alpha(alpha)
    theta_1, theta_2, theta_3 = residual_variance_moments(
        explained_variance, noise_variance, n_samples, n_features, used_components
    )
    variances = np.asarray(explained_variance, dtype=float)
    total_variance = float(np.sum(variances)) + _unfitted_component_count(
        n_samples, n_features, variances.shape[0]
    ) * float(noise_variance)
    if (
        not np.isfinite(theta_1)
        or theta_1 <= total_variance * RELATIVE_VARIANCE_TOLERANCE
    ):
        raise ValueError(
            "no residual variance remains outside the used components; "
            "reduce cumulative_explained_variance"
        )

    alpha = float(alpha)
    h0 = 1.0 - 2.0 * theta_1 * theta_3 / (3.0 * theta_2**2)
    if h0 <= 0:
        return _box_ucl(theta_1, theta_2, alpha)
    z = float(normal_distribution.isf(alpha))
    bracket = (
        z * np.sqrt(2.0 * theta_2 * h0**2) / theta_1
        + 1.0
        + theta_2 * h0 * (h0 - 1.0) / theta_1**2
    )
    if bracket <= 0:
        return _box_ucl(theta_1, theta_2, alpha)
    return theta_1 * float(bracket ** (1.0 / h0))


def spe_threshold(
    pca: PCA,
    alpha: float,
    cumulative_explained_variance: float | None,
) -> float:
    """Return the upper control limit of Q for a fitted PCA.

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
        Upper control limit of Q.

    Raises
    ------
    ValueError
        If ``alpha`` or the selector is out of range, or no residual
        component has variance.
    """
    used_components = resolve_mahalanobis_components(
        pca, cumulative_explained_variance
    )
    return spe_ucl(
        np.asarray(pca.explained_variance_, dtype=float),
        float(pca.noise_variance_),
        int(pca.n_samples_),
        int(pca.n_features_in_),
        used_components,
        alpha,
    )


def validate_spe_request(
    pca: PCA,
    config: SpeConfig,
    existing_columns: Collection[str],
) -> None:
    """Check that SPE columns can be appended before transforming.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator.
    config : SpeConfig
        Requested Q statistic settings.
    existing_columns : Collection[str]
        Columns already present in the output (input, score, and
        Mahalanobis columns).

    Raises
    ------
    ValueError
        If an SPE column collides with an existing column, the component
        selector is invalid, or no residual component has variance.
    """
    colliding = [name for name in config.column_names if name in existing_columns]
    if colliding:
        raise ValueError(f"SPE columns collide with existing columns: {colliding}")
    spe_threshold(pca, config.alpha, config.cumulative_explained_variance)


def spe_score_columns(
    pca: PCA,
    standardized_values: np.ndarray,
    scores: np.ndarray,
    config: SpeConfig,
) -> dict[str, np.ndarray]:
    """Return the Q, UCL, and exceeds-UCL columns for transformed rows.

    Parameters
    ----------
    pca : PCA
        Fitted scikit-learn PCA estimator that produced ``scores``.
    standardized_values : np.ndarray
        Preprocessed feature matrix passed to ``pca.transform``.
    scores : np.ndarray
        PCA scores of ``standardized_values``.
    config : SpeConfig
        Q statistic settings, including the output column names.

    Returns
    -------
    dict[str, np.ndarray]
        Column name to values, in the order Q, UCL, exceeds-UCL.
    """
    used_components = resolve_mahalanobis_components(
        pca, config.cumulative_explained_variance
    )
    q = spe(standardized_values, scores, pca, used_components)
    ucl = spe_threshold(pca, config.alpha, config.cumulative_explained_variance)
    return {
        config.spe_column: q,
        config.ucl_column: np.full(q.shape[0], ucl),
        config.exceeds_ucl_column: q > ucl,
    }
