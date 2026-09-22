"""Tests for PCA feature engineering."""

import numpy as np
import polars as pl
import polars.testing
import pytest

from flat_pca.feature_engineering.pca import PcaModel, fit_pca, transform_pca
from flat_pca.feature_engineering.pca.impute import DEFAULT_KMEANS_N_CLUSTERS


def test_none_scaling_leaves_centering_to_sklearn_pca() -> None:
    """Fit PCA on unscaled values and preserve source feature columns."""
    frame = pl.DataFrame(
        {
            "filename": ["a", "b", "c"],
            "feature_a": [1.0, 2.0, 4.0],
            "feature_b": [10.0, 30.0, 90.0],
        }
    )
    columns = ["feature_a", "feature_b"]

    model = fit_pca(
        frame.lazy(),
        columns,
        n_component=1,
        max_n_component=None,
        scaling_strategy="none",
    )
    result = transform_pca(frame.lazy(), model).collect()

    np.testing.assert_allclose(model.pca.mean_, [7.0 / 3.0, 130.0 / 3.0])
    assert model.scaling_model.strategy == "none"
    assert result.select(columns).equals(frame.select(columns))
    assert "pca-1" in result.columns


@pytest.fixture
def fitted_model() -> PcaModel:
    """Return a two-component model fitted on three uncorrelated features.

    Returns
    -------
    PcaModel
        Model fitted with ``scaling_strategy="none"`` over five samples, so
        every component and coefficient is deterministic.
    """
    frame = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [5.0, 3.0, 4.0, 1.0, 2.0],
            "feature_c": [2.0, 2.0, 2.0, 8.0, 8.0],
        }
    )
    return fit_pca(
        frame.lazy(),
        ["feature_a", "feature_b", "feature_c"],
        n_component=2,
        max_n_component=None,
        scaling_strategy="none",
    )


class TestGetComponentCoefficients:
    """Tests for ``PcaModel.get_component_coefficients``."""

    def test_returns_coefficients_sorted_by_abs_magnitude(
        self, fitted_model: PcaModel
    ) -> None:
        """Return one row per feature, sorted by descending |coefficient|."""
        result = fitted_model.get_component_coefficients(1)

        assert set(result["column"]) == {"feature_a", "feature_b", "feature_c"}
        abs_coefficients = result["abs_coefficient"].to_list()
        assert abs_coefficients == sorted(abs_coefficients, reverse=True)
        np.testing.assert_allclose(
            result["abs_coefficient"].to_numpy(), result["coefficient"].abs().to_numpy()
        )

    @pytest.mark.parametrize("component", [0, -1])
    def test_rejects_non_positive_component(
        self, fitted_model: PcaModel, component: int
    ) -> None:
        """Reject a component number that is not a positive integer."""
        with pytest.raises(ValueError, match="greater than 0"):
            fitted_model.get_component_coefficients(component)

    def test_rejects_component_beyond_fitted_count(
        self, fitted_model: PcaModel
    ) -> None:
        """Reject a component number beyond ``n_component``."""
        with pytest.raises(ValueError, match="less than or equal to n_component"):
            fitted_model.get_component_coefficients(fitted_model.n_component + 1)


class TestGetFeatureContributionRanking:
    """Tests for ``PcaModel.get_feature_contribution_ranking``."""

    def test_defaults_to_every_fitted_component(self, fitted_model: PcaModel) -> None:
        """Use every fitted component when no selector is given."""
        result = fitted_model.get_feature_contribution_ranking()

        assert result["used_components"].unique().to_list() == [
            fitted_model.n_component
        ]
        assert set(result["feature"]) == {"feature_a", "feature_b", "feature_c"}
        contributions = result["contribution"].to_list()
        assert contributions == sorted(contributions, reverse=True)

    def test_int_selector_caps_component_count(self, fitted_model: PcaModel) -> None:
        """Treat an int selector as a leading-component-count cap."""
        result = fitted_model.get_feature_contribution_ranking(1)

        assert result["used_components"].unique().to_list() == [1]

    def test_float_selector_uses_variance_threshold(
        self, fitted_model: PcaModel
    ) -> None:
        """Treat a float selector as a cumulative explained-variance threshold."""
        result = fitted_model.get_feature_contribution_ranking(1.0)

        assert result["used_components"][0] <= fitted_model.n_component
        assert result["used_cumulative_explained_variance"][0] <= 1.0 + 1e-9

    def test_include_component_breakdown_adds_per_component_columns(
        self, fitted_model: PcaModel
    ) -> None:
        """Append one contribution column per used component when requested."""
        result = fitted_model.get_feature_contribution_ranking(
            include_component_breakdown=True
        )

        assert "contribution_pc1" in result.columns
        assert "contribution_pc2" in result.columns

    @pytest.mark.parametrize(
        "selector",
        [True, 0, -1, 0.0, 1.5, "1"],
    )
    def test_rejects_invalid_selector(
        self, fitted_model: PcaModel, selector: object
    ) -> None:
        """Reject a selector outside the documented None/int/float contract."""
        with pytest.raises(ValueError, match="cumulative_explained_variance"):
            fitted_model.get_feature_contribution_ranking(selector)  # type: ignore[arg-type]


class TestTransformPayloadRoundTrip:
    """Tests for payload and JSON serialization round trips."""

    def test_from_transform_payload_reproduces_transform(
        self, fitted_model: PcaModel
    ) -> None:
        """Restore a model from its payload and reproduce identical scores."""
        frame = pl.DataFrame(
            {
                "feature_a": [1.5, 2.5],
                "feature_b": [4.0, 3.0],
                "feature_c": [3.0, 6.0],
            }
        )
        expected = transform_pca(frame.lazy(), fitted_model).collect()

        restored = PcaModel.from_transform_payload(fitted_model.to_transform_payload())
        actual = transform_pca(frame.lazy(), restored).collect()

        assert actual.equals(expected)

    def test_from_transform_json_round_trip_reproduces_transform(
        self, fitted_model: PcaModel
    ) -> None:
        """Restore a model from its JSON form and reproduce identical scores."""
        frame = pl.DataFrame(
            {
                "feature_a": [1.5, 2.5],
                "feature_b": [4.0, 3.0],
                "feature_c": [3.0, 6.0],
            }
        )
        expected = transform_pca(frame.lazy(), fitted_model).collect()

        restored = PcaModel.from_transform_json(fitted_model.to_transform_json())
        actual = transform_pca(frame.lazy(), restored).collect()

        assert actual.equals(expected)

    def test_restored_model_preserves_analysis_methods(
        self, fitted_model: PcaModel
    ) -> None:
        """Restore a model whose analysis methods still match the original."""
        restored = PcaModel.from_transform_payload(fitted_model.to_transform_payload())

        pl.testing.assert_frame_equal(
            restored.get_component_coefficients(1),
            fitted_model.get_component_coefficients(1),
        )
        pl.testing.assert_frame_equal(
            restored.get_feature_contribution_ranking(),
            fitted_model.get_feature_contribution_ranking(),
        )


class TestKmeansImputeStrategy:
    """Tests for ``impute_strategy="kmeans"`` in ``fit_pca``/``transform_pca``."""

    @staticmethod
    def _frame_with_missing_values() -> pl.DataFrame:
        """Return a frame with two well-separated clusters and one gap.

        Returns
        -------
        pl.DataFrame
            Six rows over ``feature_a``/``feature_b``, with one row missing
            ``feature_a``.
        """
        return pl.DataFrame(
            {
                "feature_a": [0.0, 0.0, 0.1, 10.0, 10.0, None],
                "feature_b": [0.0, 0.1, 0.0, 10.0, 9.9, 0.05],
            }
        )

    def test_fits_and_fills_missing_values_with_nearest_centroid(self) -> None:
        """Fit kmeans imputation and fill the missing feature from its cluster."""
        frame = self._frame_with_missing_values()

        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="kmeans",
            impute_kmeans_n_clusters=2,
            scaling_strategy="none",
        )

        assert model.impute_strategy == "kmeans"
        assert model.impute_kmeans_n_clusters == 2
        assert len(model.impute_kmeans_centroids) == 2
        assert model.impute_values == {}

        result = transform_pca(frame.lazy(), model).collect()
        assert result.height == 6
        assert result["feature_a"][5] == pytest.approx(0.0, abs=1.0)

    def test_uses_default_cluster_count_when_unset(self) -> None:
        """Fall back to the module default cluster count when unset."""
        frame = pl.DataFrame(
            {
                "feature_a": list(range(20)),
                "feature_b": list(range(20)),
            }
        ).with_columns(pl.all().cast(pl.Float64))

        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="kmeans",
            scaling_strategy="none",
        )

        assert model.impute_kmeans_n_clusters == DEFAULT_KMEANS_N_CLUSTERS

    def test_clips_cluster_count_to_complete_row_count(self) -> None:
        """Clip the requested cluster count to the available complete rows."""
        frame = pl.DataFrame(
            {
                "feature_a": [1.0, 2.0, None],
                "feature_b": [1.0, 2.0, 3.0],
            }
        )

        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="kmeans",
            impute_kmeans_n_clusters=10,
            scaling_strategy="none",
        )

        assert model.impute_kmeans_n_clusters == 2

    def test_transform_reuses_fitted_centroids_without_refitting(self) -> None:
        """Fill a transform-time gap from the fitted centroids, not a refit."""
        frame = self._frame_with_missing_values()
        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="kmeans",
            impute_kmeans_n_clusters=2,
            scaling_strategy="none",
        )

        new_frame = pl.DataFrame(
            {"feature_a": [None], "feature_b": [9.95]}
        )
        result = transform_pca(new_frame.lazy(), model).collect()

        assert result["feature_a"][0] == pytest.approx(10.0, abs=1.0)

    def test_rejects_all_missing_rows(self) -> None:
        """Reject fitting when every row has a missing feature value."""
        frame = pl.DataFrame({"feature_a": [None, 1.0], "feature_b": [1.0, None]})

        with pytest.raises(ValueError, match="at least one row"):
            fit_pca(
                frame.lazy(),
                ["feature_a", "feature_b"],
                n_component=1,
                max_n_component=None,
                impute_strategy="kmeans",
                scaling_strategy="none",
            )

    @pytest.mark.parametrize("n_clusters", [0, -1, True, "2", 1.5])
    def test_rejects_invalid_cluster_count(self, n_clusters: object) -> None:
        """Reject a non-positive, boolean, or non-integer cluster count."""
        frame = self._frame_with_missing_values()

        with pytest.raises(ValueError, match="impute_kmeans_n_clusters"):
            fit_pca(
                frame.lazy(),
                ["feature_a", "feature_b"],
                n_component=1,
                max_n_component=None,
                impute_strategy="kmeans",
                impute_kmeans_n_clusters=n_clusters,  # type: ignore[arg-type]
                scaling_strategy="none",
            )

    def test_impute_kmeans_n_clusters_is_keyword_only(self) -> None:
        """Reject a positional ``impute_kmeans_n_clusters`` call.

        ``impute_kmeans_n_clusters`` is keyword-only precisely so that it
        can be added without shifting the meaning of any existing
        positional ``fit_pca`` argument (for example ``outlier_strategy``,
        which sits at the same position ``impute_kmeans_n_clusters`` would
        otherwise occupy).
        """
        with pytest.raises(TypeError):
            fit_pca(  # type: ignore[misc]
                pl.DataFrame({"a": [1.0]}).lazy(),
                ["a"],
                1,
                None,
                "drop",
                None,
                1.5,
                "none",
                2,
            )

    def test_existing_positional_call_still_works(self) -> None:
        """Preserve the pre-kmeans positional argument order of ``fit_pca``.

        Regression test: a caller using the documented positional order up
        to ``scaling_strategy`` must keep working unchanged after adding
        ``impute_kmeans_n_clusters``.
        """
        frame = pl.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0], "feature_b": [3.0, 2.0, 1.0]}
        )

        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            1,
            None,
            "median",
            "winsorize",
            1.5,
            "none",
        )

        assert model.impute_strategy == "median"
        assert model.outlier_strategy == "winsorize"

    def test_rejects_cluster_count_with_non_kmeans_strategy(self) -> None:
        """Reject specifying a cluster count outside kmeans imputation."""
        frame = self._frame_with_missing_values()

        with pytest.raises(ValueError, match="impute_kmeans_n_clusters"):
            fit_pca(
                frame.lazy(),
                ["feature_a", "feature_b"],
                n_component=1,
                max_n_component=None,
                impute_strategy="median",
                impute_kmeans_n_clusters=2,
                scaling_strategy="none",
            )

    def test_payload_round_trip_reproduces_kmeans_transform(self) -> None:
        """Restore a kmeans-fitted model from its payload and reproduce fills."""
        frame = self._frame_with_missing_values()
        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="kmeans",
            impute_kmeans_n_clusters=2,
            scaling_strategy="none",
        )
        expected = transform_pca(frame.lazy(), model).collect()

        restored = PcaModel.from_transform_payload(model.to_transform_payload())
        actual = transform_pca(frame.lazy(), restored).collect()

        assert restored.impute_kmeans_n_clusters == model.impute_kmeans_n_clusters
        assert restored.impute_kmeans_centroids == model.impute_kmeans_centroids
        assert actual.equals(expected)

    def test_payload_without_kmeans_keys_restores_as_unused(self) -> None:
        """Restore a payload written before kmeans imputation existed."""
        frame = pl.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0], "feature_b": [3.0, 2.0, 1.0]}
        )
        model = fit_pca(
            frame.lazy(),
            ["feature_a", "feature_b"],
            n_component=1,
            max_n_component=None,
            impute_strategy="median",
            scaling_strategy="none",
        )
        payload = model.to_transform_payload()
        del payload["impute_kmeans_n_clusters"]
        del payload["impute_kmeans_centroids"]

        restored = PcaModel.from_transform_payload(payload)

        assert restored.impute_kmeans_n_clusters is None
        assert restored.impute_kmeans_centroids == []
