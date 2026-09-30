"""Tests for Mahalanobis distance and control limit in PCA score space."""

import numpy as np
import polars as pl
import polars.testing
import pytest
from scipy.stats import f as f_distribution

from flat_pca.feature_engineering import append_pca_scores
from flat_pca.feature_engineering.pca import (
    MahalanobisConfig,
    PcaModel,
    fit_and_transform_pca,
    fit_pca,
    transform_pca,
)
from flat_pca.feature_engineering.pca.mahalanobis import (
    mahalanobis_distance_sq,
    mahalanobis_ucl,
)

FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d"]
MAHALANOBIS_COLUMNS = [
    "mahalanobis_sq",
    "mahalanobis_ucl",
    "mahalanobis_exceeds_ucl",
]


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
            [3.0, 0.5, 0.0, 0.2],
            [0.0, 2.0, 0.3, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 0.5],
        ]
    )
    values = rng.normal(size=(n_rows, len(FEATURES))) @ mixing
    return pl.DataFrame(
        {
            "id": list(range(n_rows)),
            **{name: values[:, index] for index, name in enumerate(FEATURES)},
        }
    )


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
    """Return unseen observations, one of them far from the training data.

    Returns
    -------
    pl.DataFrame
        Six rows of random features, the last scaled far out.
    """
    frame = _random_frame(6, seed=1)
    return frame.with_columns(
        pl.when(pl.col("id") == 5)
        .then(pl.col(name) * 20)
        .otherwise(pl.col(name))
        .alias(name)
        for name in FEATURES
    )


@pytest.fixture
def model(train_frame: pl.DataFrame) -> PcaModel:
    """Return a PCA model fitted on every feature component.

    Returns
    -------
    PcaModel
        Four-component model fitted without scaling.
    """
    return fit_pca(
        train_frame.lazy(),
        FEATURES,
        max_n_component=None,
        scaling_strategy="none",
    )


class TestMahalanobisConfig:
    """Validation of ``MahalanobisConfig``."""

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, True, "0.1"])
    def test_rejects_alpha_out_of_range(self, alpha: object) -> None:
        """Reject significance levels outside the open interval (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            MahalanobisConfig(alpha=alpha)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field", ["distance_column", "ucl_column", "exceeds_ucl_column"]
    )
    def test_rejects_empty_column_name(self, field: str) -> None:
        """Reject an empty output column name."""
        with pytest.raises(ValueError, match="non-empty"):
            MahalanobisConfig(**{field: ""})

    def test_rejects_duplicate_column_names(self) -> None:
        """Reject output column names that are not distinct."""
        with pytest.raises(ValueError, match="distinct"):
            MahalanobisConfig(ucl_column="mahalanobis_sq")


class TestMahalanobisFunctions:
    """Pure distance and control-limit computations."""

    def test_distance_matches_score_covariance(
        self, model: PcaModel, train_frame: pl.DataFrame, new_frame: pl.DataFrame
    ) -> None:
        """Match the direct distance using the covariance of training scores."""
        train_scores = model.pca.transform(train_frame.select(FEATURES).to_numpy())
        new_scores = model.pca.transform(new_frame.select(FEATURES).to_numpy())
        used_components = 3

        used_train = train_scores[:, :used_components]
        inverse_covariance = np.linalg.inv(np.cov(used_train, rowvar=False))
        used_new = new_scores[:, :used_components]
        expected = np.einsum("ij,jk,ik->i", used_new, inverse_covariance, used_new)

        actual = mahalanobis_distance_sq(
            new_scores, model.pca.explained_variance_, used_components
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_ucl_matches_f_distribution_formula(self) -> None:
        """Match the F-distribution formula for new observations."""
        n_samples, used_components, alpha = 30, 3, 0.05
        expected = (
            used_components
            * (n_samples + 1)
            * (n_samples - 1)
            / (n_samples * (n_samples - used_components))
            * f_distribution.ppf(1 - alpha, used_components, n_samples - used_components)
        )

        assert mahalanobis_ucl(n_samples, used_components, alpha) == pytest.approx(
            expected
        )

    def test_ucl_stays_finite_for_tiny_alpha(self) -> None:
        """Keep the limit finite when ``1 - alpha`` rounds to 1."""
        n_samples, used_components, alpha = 30, 3, 1e-20
        scale = (
            used_components
            * (n_samples + 1)
            * (n_samples - 1)
            / (n_samples * (n_samples - used_components))
        )

        actual = mahalanobis_ucl(n_samples, used_components, alpha)

        assert np.isfinite(actual)
        # The survival function is evaluated without ``1 - alpha``, so it
        # recovers the tiny upper-tail probability accurately.
        assert f_distribution.sf(
            actual / scale, used_components, n_samples - used_components
        ) == pytest.approx(alpha)
        assert actual > mahalanobis_ucl(n_samples, used_components, 1e-10)

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.9, 0.999999999])
    @pytest.mark.parametrize(("n_samples", "used_components"), [(30, 1), (30, 3)])
    def test_ucl_matches_ppf_for_large_alpha(
        self, n_samples: int, used_components: int, alpha: float
    ) -> None:
        """Keep the limit positive and exact when ``alpha`` is close to 1."""
        dfd = n_samples - used_components
        expected = (
            used_components
            * (n_samples + 1)
            * (n_samples - 1)
            / (n_samples * dfd)
            * f_distribution.ppf(1 - alpha, used_components, dfd)
        )

        actual = mahalanobis_ucl(n_samples, used_components, alpha)

        assert actual > 0
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_ucl_rejects_components_not_below_samples(self) -> None:
        """Reject a component count that leaves no F degrees of freedom."""
        with pytest.raises(ValueError, match="less than the fitted sample count"):
            mahalanobis_ucl(5, 5, 0.01)

    def test_distance_rejects_near_zero_variance(self) -> None:
        """Reject a used component whose variance is effectively zero."""
        scores = np.ones((2, 2))
        with pytest.raises(ValueError, match="near-zero variance"):
            mahalanobis_distance_sq(scores, np.array([1.0, 1e-20]), 2)


class TestTransformWithMahalanobis:
    """Mahalanobis columns appended by the transform functions."""

    def test_default_output_is_unchanged(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Leave the output unchanged when no config is given."""
        expected = transform_pca(new_frame.lazy(), model).collect()
        actual = transform_pca(new_frame.lazy(), model, mahalanobis=None).collect()

        polars.testing.assert_frame_equal(actual, expected)
        assert not set(MAHALANOBIS_COLUMNS) & set(actual.columns)

    def test_appends_distance_ucl_and_flag(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Append the three columns after the score columns."""
        config = MahalanobisConfig(alpha=0.05)
        result = transform_pca(new_frame.lazy(), model, mahalanobis=config).collect()

        assert result.columns == [
            *new_frame.columns,
            *model.pca_column_names,
            *MAHALANOBIS_COLUMNS,
        ]
        assert result.schema["mahalanobis_exceeds_ucl"] == pl.Boolean
        used_components = int(
            np.searchsorted(np.cumsum(model.pca.explained_variance_ratio_), 0.9) + 1
        )
        scores = result.select(model.pca_column_names).to_numpy()
        expected_distance = np.sum(
            scores[:, :used_components] ** 2
            / model.pca.explained_variance_[:used_components],
            axis=1,
        )
        expected_ucl = model.get_mahalanobis_threshold(alpha=0.05)

        np.testing.assert_allclose(result["mahalanobis_sq"].to_numpy(), expected_distance)
        assert result["mahalanobis_ucl"].to_list() == [expected_ucl] * result.height
        assert result["mahalanobis_exceeds_ucl"].to_list() == list(
            expected_distance > expected_ucl
        )
        assert result["mahalanobis_exceeds_ucl"][-1] is True

    def test_uses_custom_column_names(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Name the appended columns after the config."""
        config = MahalanobisConfig(
            distance_column="t2",
            ucl_column="t2_limit",
            exceeds_ucl_column="t2_alarm",
        )
        result = transform_pca(new_frame.lazy(), model, mahalanobis=config).collect()

        assert result.columns[-3:] == ["t2", "t2_limit", "t2_alarm"]

    def test_fit_and_transform_forwards_config(self, train_frame: pl.DataFrame) -> None:
        """Forward the config through ``fit_and_transform_pca``."""
        result = fit_and_transform_pca(
            train_frame.lazy(),
            FEATURES,
            max_n_component=None,
            scaling_strategy="none",
            mahalanobis=MahalanobisConfig(),
        ).collect()

        assert result.columns[-3:] == MAHALANOBIS_COLUMNS

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
        result = append_pca_scores(
            model, flattened.lazy(), mahalanobis=MahalanobisConfig()
        ).collect()

        assert result.columns[-3:] == MAHALANOBIS_COLUMNS

    @pytest.mark.parametrize("colliding", ["feature_a", "id", "pca-1"])
    def test_rejects_colliding_column_name(
        self, model: PcaModel, new_frame: pl.DataFrame, colliding: str
    ) -> None:
        """Reject a column name already used by the input or score columns."""
        config = MahalanobisConfig(distance_column=colliding)
        with pytest.raises(ValueError, match="collide"):
            transform_pca(new_frame.lazy(), model, mahalanobis=config)

    def test_rejects_components_not_below_samples(self) -> None:
        """Reject selecting as many components as fitted samples."""
        frame = _random_frame(4, seed=2)
        model = fit_pca(
            frame.lazy(), FEATURES, max_n_component=None, scaling_strategy="none"
        )
        config = MahalanobisConfig(cumulative_explained_variance=4)

        with pytest.raises(ValueError, match="less than the fitted sample count"):
            transform_pca(frame.lazy(), model, mahalanobis=config)

    def test_rejects_near_zero_variance_component(self) -> None:
        """Reject a used component whose variance is effectively zero."""
        frame, model = _rank_one_model()
        config = MahalanobisConfig(cumulative_explained_variance=2)

        with pytest.raises(ValueError, match="near-zero variance"):
            transform_pca(frame.lazy(), model, mahalanobis=config)


def _rank_one_model() -> tuple[pl.DataFrame, PcaModel]:
    """Return rank-one data and a two-component model fitted on it.

    Returns
    -------
    tuple[pl.DataFrame, PcaModel]
        Collinear features and a model whose second component has
        effectively zero variance.
    """
    base = np.linspace(1.0, 10.0, 8)
    frame = pl.DataFrame({"feature_a": base, "feature_b": 2 * base, "feature_c": -base})
    model = fit_pca(
        frame.lazy(),
        ["feature_a", "feature_b", "feature_c"],
        n_component=2,
        max_n_component=None,
        scaling_strategy="none",
    )
    return frame, model


class TestModelThreshold:
    """``PcaModel.get_mahalanobis_threshold`` and restored models."""

    def test_threshold_matches_formula(self, model: PcaModel) -> None:
        """Return the UCL of the resolved component count and sample count."""
        expected = mahalanobis_ucl(30, 2, 0.01)

        assert model.get_mahalanobis_threshold(
            cumulative_explained_variance=2
        ) == pytest.approx(expected)

    def test_threshold_rejects_alpha_out_of_range(self, model: PcaModel) -> None:
        """Reject a significance level outside (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            model.get_mahalanobis_threshold(alpha=1.0)

    def test_threshold_rejects_near_zero_variance_component(self) -> None:
        """Reject the same selector that ``transform_pca`` rejects."""
        _, model = _rank_one_model()

        with pytest.raises(ValueError, match="near-zero variance"):
            model.get_mahalanobis_threshold(cumulative_explained_variance=2)

    def test_json_round_trip_preserves_distance_and_ucl(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Reproduce distances and UCL after a JSON round trip."""
        config = MahalanobisConfig()
        expected = transform_pca(new_frame.lazy(), model, mahalanobis=config).collect()

        restored = PcaModel.from_transform_json(model.to_transform_json())
        actual = transform_pca(new_frame.lazy(), restored, mahalanobis=config).collect()

        polars.testing.assert_frame_equal(actual, expected)
        assert restored.get_mahalanobis_threshold() == model.get_mahalanobis_threshold()

    def test_whitened_model_gives_same_distance(
        self, model: PcaModel, new_frame: pl.DataFrame
    ) -> None:
        """Compute the same distance from a model restored with whitening."""
        payload = model.to_transform_payload()
        payload["pca"]["whiten"] = True  # type: ignore[index]
        whitened = PcaModel.from_transform_payload(payload)
        config = MahalanobisConfig()

        expected = transform_pca(new_frame.lazy(), model, mahalanobis=config).collect()
        actual = transform_pca(new_frame.lazy(), whitened, mahalanobis=config).collect()

        np.testing.assert_allclose(
            actual["mahalanobis_sq"].to_numpy(),
            expected["mahalanobis_sq"].to_numpy(),
        )
