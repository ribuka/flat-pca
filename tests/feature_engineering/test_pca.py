"""Tests for PCA feature engineering."""

import numpy as np
import polars as pl
import polars.testing
import pytest

from flat_pca.feature_engineering.pca import PcaModel, fit_pca, transform_pca


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
