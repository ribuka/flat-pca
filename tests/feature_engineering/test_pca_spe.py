"""Tests for the Q statistic (SPE) and its control limit."""

import numpy as np
import polars as pl
import polars.testing
import pytest
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import norm as normal_distribution

from flat_pca.feature_engineering import append_pca_scores
from flat_pca.feature_engineering.pca import (
    MahalanobisConfig,
    PcaModel,
    SpeConfig,
    fit_and_transform_pca,
    fit_pca,
    transform_pca,
)
from flat_pca.feature_engineering.pca.spe import spe, spe_ucl

FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]
SPE_COLUMNS = ["spe", "spe_ucl", "spe_exceeds_ucl"]


def _random_frame(n_rows: int, seed: int) -> pl.DataFrame:
    """Return a frame of correlated random features.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    seed : int
        Seed of the random generator.

    Returns
    -------
    pl.DataFrame
        Frame with an ``id`` column and the ``FEATURES`` columns.
    """
    rng = np.random.default_rng(seed)
    mixing = np.array(
        [
            [3.0, 0.5, 0.0, 0.2, 0.1],
            [0.0, 2.0, 0.3, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.3],
        ]
    )
    values = rng.normal(size=(n_rows, len(FEATURES))) @ mixing
    return pl.DataFrame(
        {
            "id": list(range(n_rows)),
            **{name: values[:, index] for index, name in enumerate(FEATURES)},
        }
    )


def _jackson_mudholkar(residual_variances: np.ndarray, alpha: float) -> float:
    """Return the Jackson-Mudholkar limit computed straight from the formula.

    Parameters
    ----------
    residual_variances : np.ndarray
        Variances of the residual components.
    alpha : float
        Significance level.

    Returns
    -------
    float
        Upper control limit of Q.
    """
    theta_1, theta_2, theta_3 = (
        np.sum(residual_variances**power) for power in (1, 2, 3)
    )
    h0 = 1 - 2 * theta_1 * theta_3 / (3 * theta_2**2)
    z = normal_distribution.ppf(1 - alpha)
    return theta_1 * (
        z * np.sqrt(2 * theta_2 * h0**2) / theta_1
        + 1
        + theta_2 * h0 * (h0 - 1) / theta_1**2
    ) ** (1 / h0)


@pytest.fixture
def train_frame() -> pl.DataFrame:
    """Return the training frame.

    Returns
    -------
    pl.DataFrame
        Thirty rows of correlated random features.
    """
    return _random_frame(30, seed=0)


@pytest.fixture
def new_frame() -> pl.DataFrame:
    """Return unseen observations, one of them off the training subspace.

    Returns
    -------
    pl.DataFrame
        Six rows of random features; the last has a large ``feature_e``,
        the direction with the smallest training variance.
    """
    frame = _random_frame(6, seed=1)
    return frame.with_columns(
        pl.when(pl.col("id") == 5)
        .then(pl.col("feature_e") + 20)
        .otherwise(pl.col("feature_e"))
        .alias("feature_e")
    )


@pytest.fixture
def model(train_frame: pl.DataFrame) -> PcaModel:
    """Return a PCA model fitted on every feature component.

    Returns
    -------
    PcaModel
        Five-component model fitted without scaling.
    """
    return fit_pca(
        train_frame.lazy(),
        FEATURES,
        max_n_component=None,
        scaling_strategy="none",
    )


@pytest.fixture
def truncated_model(train_frame: pl.DataFrame) -> PcaModel:
    """Return a PCA model fitted on only the first three components.

    Returns
    -------
    PcaModel
        Three-component model whose remaining variance is kept only as
        ``noise_variance_``.
    """
    return fit_pca(
        train_frame.lazy(),
        FEATURES,
        max_n_component=3,
        scaling_strategy="none",
    )


class TestSpeConfig:
    """Validation of ``SpeConfig``."""

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, True, "0.1"])
    def test_rejects_alpha_out_of_range(self, alpha: object) -> None:
        """Reject significance levels outside the open interval (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            SpeConfig(alpha=alpha)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["spe_column", "ucl_column", "exceeds_ucl_column"])
    def test_rejects_empty_column_name(self, field: str) -> None:
        """Reject an empty output column name."""
        with pytest.raises(ValueError, match="non-empty"):
            SpeConfig(**{field: ""})

    def test_rejects_duplicate_column_names(self) -> None:
        """Reject output column names that are not distinct."""
        with pytest.raises(ValueError, match="distinct"):
            SpeConfig(ucl_column="spe")


class TestSpeFunctions:
    """Pure Q statistic and control-limit computations."""

    @pytest.mark.parametrize("used_components", [1, 2, 3])
    def test_spe_matches_leading_component_reconstruction(
        self, model: PcaModel, new_frame: pl.DataFrame, used_components: int
    ) -> None:
        """Match the error of the reconstruction from the leading components."""
        values = new_frame.select(FEATURES).to_numpy()
        scores = model.pca.transform(values)
        centered = values - model.pca.mean_
        loadings = model.pca.components_[:used_components]
        residual = centered - centered @ loadings.T @ loadings
        expected = np.sum(residual**2, axis=1)

        actual = spe(values, scores, model.pca, used_components)

        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_spe_with_every_component_matches_inverse_transform(
        self, truncated_model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Match ``pca.inverse_transform`` when every fitted component is used."""
        values = new_frame.select(FEATURES).to_numpy()
        scores = truncated_model.pca.transform(values)
        expected = np.sum(
            (values - truncated_model.pca.inverse_transform(scores)) ** 2, axis=1
        )

        actual = spe(values, scores, truncated_model.pca, 3)

        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_ucl_matches_jackson_mudholkar_formula(self) -> None:
        """Match the formula over the residual component variances."""
        explained_variance = np.array([9.0, 4.0, 1.0, 0.5, 0.3])
        expected = _jackson_mudholkar(explained_variance[2:], 0.05)

        actual = spe_ucl(explained_variance, 0.0, 30, 5, 2, 0.05)

        assert actual == pytest.approx(expected, rel=1e-12)

    def test_ucl_gives_unfitted_components_the_noise_variance(self) -> None:
        """Count each unfitted component up to the rank bound as the noise variance."""
        explained_variance = np.array([9.0, 4.0, 1.0])
        noise_variance = 0.2
        # rank bound min(30, 6) = 6, so three components were not fitted.
        residual = np.array([4.0, 1.0, noise_variance, noise_variance, noise_variance])
        expected = _jackson_mudholkar(residual, 0.01)

        actual = spe_ucl(explained_variance, noise_variance, 30, 6, 1, 0.01)

        assert actual == pytest.approx(expected, rel=1e-12)

    def test_ucl_stays_finite_for_tiny_alpha(self) -> None:
        """Keep the limit finite when ``1 - alpha`` rounds to 1."""
        explained_variance = np.array([9.0, 4.0, 1.0, 0.5, 0.3])

        actual = spe_ucl(explained_variance, 0.0, 30, 5, 2, 1e-20)

        assert np.isfinite(actual)
        assert actual > spe_ucl(explained_variance, 0.0, 30, 5, 2, 1e-10)

    def test_ucl_falls_back_to_box_when_h0_is_not_positive(self) -> None:
        """Use ``g chi2(h)`` when one residual variance dominates many tiny ones."""
        explained_variance = np.array([10.0, 1.0, *[0.01] * 10_000])
        theta_1 = float(np.sum(explained_variance[1:]))
        theta_2 = float(np.sum(explained_variance[1:] ** 2))
        theta_3 = float(np.sum(explained_variance[1:] ** 3))
        assert 1 - 2 * theta_1 * theta_3 / (3 * theta_2**2) <= 0
        expected = theta_2 / theta_1 * chi2_distribution.ppf(
            0.99, theta_1**2 / theta_2
        )

        actual = spe_ucl(explained_variance, 0.0, 20_000, 10_002, 1, 0.01)

        assert actual == pytest.approx(expected, rel=1e-12)

    def test_ucl_rejects_no_residual_variance(self) -> None:
        """Reject keeping every component up to the rank bound."""
        with pytest.raises(ValueError, match="no residual variance"):
            spe_ucl(np.array([9.0, 4.0, 1.0]), 0.0, 30, 3, 3, 0.01)

    def test_ucl_rejects_alpha_out_of_range(self) -> None:
        """Reject a significance level outside (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            spe_ucl(np.array([9.0, 4.0, 1.0]), 0.0, 30, 3, 1, 1.0)


class TestTransformWithSpe:
    """SPE columns appended by the transform functions."""

    def test_default_output_is_unchanged(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Leave the output unchanged when no config is given."""
        expected = transform_pca(new_frame.lazy(), model).collect()
        actual = transform_pca(new_frame.lazy(), model, spe=None).collect()

        polars.testing.assert_frame_equal(actual, expected)
        assert not set(SPE_COLUMNS) & set(actual.columns)

    def test_appends_spe_ucl_and_flag(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Append the three columns after the score columns."""
        config = SpeConfig(cumulative_explained_variance=2, alpha=0.05)
        result = transform_pca(new_frame.lazy(), model, spe=config).collect()

        assert result.columns == [
            *new_frame.columns,
            *model.pca_column_names,
            *SPE_COLUMNS,
        ]
        assert result.schema["spe_exceeds_ucl"] == pl.Boolean
        values = new_frame.select(FEATURES).to_numpy()
        expected_spe = spe(values, model.pca.transform(values), model.pca, 2)
        expected_ucl = model.get_spe_threshold(
            alpha=0.05, cumulative_explained_variance=2
        )

        np.testing.assert_allclose(result["spe"].to_numpy(), expected_spe)
        assert result["spe_ucl"].to_list() == [expected_ucl] * result.height
        assert result["spe_exceeds_ucl"].to_list() == list(expected_spe > expected_ucl)
        assert result["spe_exceeds_ucl"][-1] is True

    def test_uses_preprocessed_values(
        self, train_frame: pl.DataFrame, new_frame: pl.DataFrame
    ) -> None:
        """Compute Q from the scaled values on the Polars preprocessing path."""
        model = fit_pca(
            train_frame.lazy(), FEATURES, max_n_component=None, scaling_strategy="robust"
        )
        result = transform_pca(new_frame.lazy(), model, spe=SpeConfig()).collect()
        scaled = result.select(FEATURES).to_numpy()
        used_components = int(
            np.searchsorted(np.cumsum(model.pca.explained_variance_ratio_), 0.9) + 1
        )

        expected = spe(scaled, model.pca.transform(scaled), model.pca, used_components)

        np.testing.assert_allclose(result["spe"].to_numpy(), expected)

    def test_appends_after_mahalanobis_columns(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Place the SPE columns after the Mahalanobis columns."""
        result = transform_pca(
            new_frame.lazy(),
            model,
            mahalanobis=MahalanobisConfig(),
            spe=SpeConfig(),
        ).collect()

        assert result.columns[-6:] == [
            *MahalanobisConfig().column_names,
            *SPE_COLUMNS,
        ]

    def test_uses_custom_column_names(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Name the appended columns after the config."""
        config = SpeConfig(spe_column="q", ucl_column="q_limit", exceeds_ucl_column="q_alarm")
        result = transform_pca(new_frame.lazy(), model, spe=config).collect()

        assert result.columns[-3:] == ["q", "q_limit", "q_alarm"]

    def test_fit_and_transform_forwards_config(self, train_frame: pl.DataFrame) -> None:
        """Forward the config through ``fit_and_transform_pca``."""
        result = fit_and_transform_pca(
            train_frame.lazy(),
            FEATURES,
            max_n_component=None,
            scaling_strategy="none",
            spe=SpeConfig(),
        ).collect()

        assert result.columns[-3:] == SPE_COLUMNS

    def test_append_pca_scores_forwards_config(
        self, train_frame: pl.DataFrame
    ) -> None:
        """Forward the config through ``append_pca_scores``."""
        flattened = train_frame.rename({"id": "source"}).with_columns(
            pl.col("source").cast(pl.String)
        )
        model = fit_pca(
            flattened.lazy(),
            FEATURES,
            max_n_component=None,
            scaling_strategy="none",
        )
        result = append_pca_scores(model, flattened.lazy(), spe=SpeConfig()).collect()

        assert result.columns[-3:] == SPE_COLUMNS

    @pytest.mark.parametrize("colliding", ["feature_a", "id", "pca-1"])
    def test_rejects_colliding_column_name(
        self, model: PcaModel, new_frame: pl.DataFrame, colliding: str
    ) -> None:
        """Reject a column name already used by the input or score columns."""
        config = SpeConfig(spe_column=colliding)
        with pytest.raises(ValueError, match="collide"):
            transform_pca(new_frame.lazy(), model, spe=config)

    def test_rejects_name_colliding_with_mahalanobis(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Reject a column name already used by the Mahalanobis columns."""
        config = SpeConfig(ucl_column=MahalanobisConfig().ucl_column)
        with pytest.raises(ValueError, match="SPE columns collide"):
            transform_pca(
                new_frame.lazy(), model, mahalanobis=MahalanobisConfig(), spe=config
            )

    def test_rejects_no_residual_variance(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Reject keeping every component when no variance is left over."""
        config = SpeConfig(cumulative_explained_variance=None)
        with pytest.raises(ValueError, match="no residual variance"):
            transform_pca(new_frame.lazy(), model, spe=config)


class TestModelThreshold:
    """``PcaModel.get_spe_threshold`` and restored models."""

    def test_threshold_matches_formula(self, truncated_model: PcaModel) -> None:
        """Return the UCL from the fitted variances and the noise variance."""
        pca = truncated_model.pca
        residual = np.array(
            [*pca.explained_variance_[1:], pca.noise_variance_, pca.noise_variance_]
        )
        expected = _jackson_mudholkar(residual, 0.01)

        assert truncated_model.get_spe_threshold(
            cumulative_explained_variance=1
        ) == pytest.approx(expected, rel=1e-12)

    def test_threshold_rejects_alpha_out_of_range(self, model: PcaModel) -> None:
        """Reject a significance level outside (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            model.get_spe_threshold(alpha=0.0)

    def test_json_round_trip_preserves_spe_and_ucl(
        self, truncated_model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Reproduce Q and UCL after a JSON round trip."""
        config = SpeConfig(cumulative_explained_variance=2)
        expected = transform_pca(new_frame.lazy(), truncated_model, spe=config).collect()

        restored = PcaModel.from_transform_json(truncated_model.to_transform_json())
        actual = transform_pca(new_frame.lazy(), restored, spe=config).collect()

        polars.testing.assert_frame_equal(actual, expected)
        assert restored.get_spe_threshold(
            cumulative_explained_variance=2
        ) == truncated_model.get_spe_threshold(cumulative_explained_variance=2)

    def test_whitened_model_gives_same_spe(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Compute the same Q from a model restored with whitening."""
        payload = model.to_transform_payload()
        payload["pca"]["whiten"] = True  # type: ignore[index]
        whitened = PcaModel.from_transform_payload(payload)
        config = SpeConfig()

        expected = transform_pca(new_frame.lazy(), model, spe=config).collect()
        actual = transform_pca(new_frame.lazy(), whitened, spe=config).collect()

        np.testing.assert_allclose(actual["spe"].to_numpy(), expected["spe"].to_numpy())
