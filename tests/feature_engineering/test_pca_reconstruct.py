"""Tests for the explained-variance table and feature reconstruction."""

from dataclasses import replace

import numpy as np
import polars as pl
import polars.testing
import pytest
from sklearn.decomposition import PCA

from flat_pca.feature_engineering.pca import PcaModel, fit_pca, transform_pca
from flat_pca.feature_engineering.pca.analysis import resolve_used_components
from flat_pca.feature_engineering.pca.reconstruct import reconstruct_standardized
from flat_pca.feature_engineering.scaling import ScalingModel, ScalingStrategy

FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]
TABLE_COLUMNS = [
    "component",
    "pca_column",
    "explained_variance",
    "explained_variance_ratio",
    "cumulative_explained_variance",
]


def _random_frame(n_rows: int, seed: int) -> pl.DataFrame:
    """Return a frame of correlated random features with an ``id`` column.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    seed : int
        Seed of the random generator.

    Returns
    -------
    pl.DataFrame
        Frame with an ``id`` column followed by the ``FEATURES`` columns.
    """
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(len(FEATURES), len(FEATURES)))
    values = rng.normal(size=(n_rows, len(FEATURES))) @ mixing + 10.0
    return pl.DataFrame(
        {
            "id": list(range(n_rows)),
            **{name: values[:, index] for index, name in enumerate(FEATURES)},
        }
    )


def _fit(frame: pl.DataFrame, scaling_strategy: ScalingStrategy) -> PcaModel:
    """Fit a model on every feature column with all components.

    Parameters
    ----------
    frame : pl.DataFrame
        Training data holding the ``FEATURES`` columns.
    scaling_strategy : ScalingStrategy
        Scaling applied before PCA.

    Returns
    -------
    PcaModel
        Fitted model with as many components as features.
    """
    return fit_pca(
        frame.lazy(),
        FEATURES,
        max_n_component=None,
        scaling_strategy=scaling_strategy,
    )


def _scores(frame: pl.DataFrame, model: PcaModel) -> pl.DataFrame:
    """Return ``id`` and the score columns of ``frame`` under ``model``.

    Parameters
    ----------
    frame : pl.DataFrame
        Data holding the model's feature columns.
    model : PcaModel
        Fitted model.

    Returns
    -------
    pl.DataFrame
        ``id`` followed by every score column.
    """
    return (
        transform_pca(frame.lazy(), model)
        .select("id", *model.pca_column_names)
        .collect()
    )


@pytest.fixture
def frame() -> pl.DataFrame:
    """Return a training frame of 30 correlated samples.

    Returns
    -------
    pl.DataFrame
        Frame with an ``id`` column and the ``FEATURES`` columns.
    """
    return _random_frame(30, seed=0)


class TestGetExplainedVarianceTable:
    """Tests for ``PcaModel.get_explained_variance_table``."""

    def test_returns_one_row_per_component(self, frame: pl.DataFrame) -> None:
        """List every fitted component with its variance and ratio."""
        model = _fit(frame, "none")

        table = model.get_explained_variance_table()

        assert table.columns == TABLE_COLUMNS
        assert table.schema == pl.Schema(
            {
                "component": pl.Int64,
                "pca_column": pl.Utf8,
                "explained_variance": pl.Float64,
                "explained_variance_ratio": pl.Float64,
                "cumulative_explained_variance": pl.Float64,
            }
        )
        assert table["component"].to_list() == [1, 2, 3, 4, 5]
        assert table["pca_column"].to_list() == list(model.pca_column_names)
        np.testing.assert_array_equal(
            table["explained_variance"].to_numpy(), model.pca.explained_variance_
        )
        np.testing.assert_array_equal(
            table["explained_variance_ratio"].to_numpy(),
            model.pca.explained_variance_ratio_,
        )
        np.testing.assert_array_equal(
            table["cumulative_explained_variance"].to_numpy(),
            np.cumsum(model.pca.explained_variance_ratio_),
        )

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9, 0.99])
    def test_cumulative_agrees_with_component_selector(
        self, frame: pl.DataFrame, threshold: float
    ) -> None:
        """Select the first component whose cumulative value reaches the threshold."""
        model = _fit(frame, "none")
        table = model.get_explained_variance_table()

        selected = resolve_used_components(
            model.pca.explained_variance_ratio_, model.n_component, threshold
        )
        first_reaching = table.filter(
            pl.col("cumulative_explained_variance") >= threshold
        )["component"][0]

        assert selected == first_reaching

    def test_unreached_threshold_selects_last_component(
        self, frame: pl.DataFrame
    ) -> None:
        """Select every fitted component when a truncated PCA misses the threshold."""
        model = fit_pca(
            frame.lazy(), FEATURES, max_n_component=2, scaling_strategy="none"
        )
        table = model.get_explained_variance_table()
        threshold = 0.999
        assert table["cumulative_explained_variance"][-1] < threshold

        selected = resolve_used_components(
            model.pca.explained_variance_ratio_, model.n_component, threshold
        )

        assert selected == table["component"][-1] == 2

    def test_rejects_inconsistent_pca_state(self, frame: pl.DataFrame) -> None:
        """Raise when the ratio length disagrees with the component count."""
        model = _fit(frame, "none")
        model.pca.explained_variance_ratio_ = model.pca.explained_variance_ratio_[:-1]

        with pytest.raises(ValueError, match="explained_variance_ratio"):
            model.get_explained_variance_table()


class TestReconstruct:
    """Tests for ``PcaModel.reconstruct``."""

    def test_all_components_match_inverse_transform(
        self, frame: pl.DataFrame
    ) -> None:
        """Agree with ``PCA.inverse_transform`` when nothing is scaled."""
        model = _fit(frame, "none")
        scores = _scores(frame, model)

        result = model.reconstruct(scores)

        expected = model.pca.inverse_transform(
            scores.select(model.pca_column_names).to_numpy()
        )
        np.testing.assert_allclose(result.select(FEATURES).to_numpy(), expected)

    @pytest.mark.parametrize("scaling_strategy", ["z-score", "minmax", "robust"])
    def test_all_components_recover_original_features(
        self, frame: pl.DataFrame, scaling_strategy: ScalingStrategy
    ) -> None:
        """Undo the scaling and recover the unimputed, unclipped input."""
        model = _fit(frame, scaling_strategy)

        result = model.reconstruct(_scores(frame, model))

        np.testing.assert_allclose(
            result.select(FEATURES).to_numpy(),
            frame.select(FEATURES).to_numpy(),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_leading_components_match_direct_computation(
        self, frame: pl.DataFrame
    ) -> None:
        """Reconstruct from the first k components only, then unscale."""
        model = _fit(frame, "z-score")
        scores = _scores(frame, model)

        result = model.reconstruct(scores, cumulative_explained_variance=2)

        score_matrix = scores.select("pca-1", "pca-2").to_numpy()
        standardized = score_matrix @ model.pca.components_[:2] + model.pca.mean_
        scales = np.array([model.scaling_model.scales[c] for c in FEATURES])
        centers = np.array([model.scaling_model.centers[c] for c in FEATURES])
        np.testing.assert_allclose(
            result.select(FEATURES).to_numpy(), standardized * scales + centers
        )

    def test_threshold_selector_uses_resolved_component_count(
        self, frame: pl.DataFrame
    ) -> None:
        """Match the integer selector that picks the same component count."""
        model = _fit(frame, "robust")
        scores = _scores(frame, model)
        count = resolve_used_components(
            model.pca.explained_variance_ratio_, model.n_component, 0.8
        )

        polars.testing.assert_frame_equal(
            model.reconstruct(scores, cumulative_explained_variance=0.8),
            model.reconstruct(scores, cumulative_explained_variance=count),
        )

    def test_keeps_other_columns_and_row_order(self, frame: pl.DataFrame) -> None:
        """Keep non-score columns first and ignore unused score columns."""
        model = _fit(frame, "none")
        scores = (
            _scores(frame, model)
            .with_columns(pl.lit("x").alias("source"))
            .reverse()
            .drop("pca-4", "pca-5")
        )

        result = model.reconstruct(scores, cumulative_explained_variance=3)

        assert result.columns == ["id", "source", *FEATURES]
        assert result["id"].to_list() == scores["id"].to_list()

    def test_replaces_feature_columns_of_transform_output(
        self, frame: pl.DataFrame
    ) -> None:
        """Replace the scaled feature columns that ``transform_pca`` keeps."""
        model = _fit(frame, "z-score")
        transformed = transform_pca(frame.lazy(), model).collect()

        result = model.reconstruct(transformed)

        assert result.columns == ["id", *FEATURES]
        np.testing.assert_allclose(
            result.select(FEATURES).to_numpy(),
            frame.select(FEATURES).to_numpy(),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_missing_scores_give_nan_rows(self, frame: pl.DataFrame) -> None:
        """Keep rows with a null or NaN used score as all-NaN rows."""
        model = _fit(frame, "none")
        scores = _scores(frame, model).head(3)
        scores = scores.with_columns(
            pl.Series("pca-1", [None, float("nan"), scores["pca-1"][2]])
        )

        result = model.reconstruct(scores)

        values = result.select(FEATURES).to_numpy()
        assert result.height == 3
        assert np.isnan(values[:2]).all()
        assert np.isfinite(values[2]).all()

    def test_whitened_pca_matches_inverse_transform(
        self, frame: pl.DataFrame
    ) -> None:
        """Multiply whitened scores back by the component standard deviation."""
        model = _fit(frame, "none")
        whitened = PCA(n_components=model.n_component, whiten=True).fit(
            frame.select(FEATURES).to_numpy()
        )
        model = replace(model, pca=whitened)
        score_matrix = whitened.transform(frame.select(FEATURES).to_numpy())
        scores = pl.DataFrame(
            {name: score_matrix[:, i] for i, name in enumerate(model.pca_column_names)}
        )

        result = model.reconstruct(scores)

        np.testing.assert_allclose(
            result.to_numpy(), whitened.inverse_transform(score_matrix)
        )

    @pytest.mark.parametrize("scaling_strategy", ["none", "z-score"])
    def test_json_round_trip_gives_same_results(
        self, frame: pl.DataFrame, scaling_strategy: ScalingStrategy
    ) -> None:
        """Reproduce the table and reconstruction from a restored model."""
        model = _fit(frame, scaling_strategy)
        restored = PcaModel.from_transform_json(model.to_transform_json())
        scores = _scores(frame, model)

        polars.testing.assert_frame_equal(
            restored.get_explained_variance_table(),
            model.get_explained_variance_table(),
        )
        polars.testing.assert_frame_equal(
            restored.reconstruct(scores, cumulative_explained_variance=0.9),
            model.reconstruct(scores, cumulative_explained_variance=0.9),
        )

    def test_rejects_missing_used_score_column(self, frame: pl.DataFrame) -> None:
        """Raise when one of the first k score columns is absent."""
        model = _fit(frame, "none")
        scores = _scores(frame, model).drop("pca-2")

        with pytest.raises(ValueError, match="pca-2"):
            model.reconstruct(scores, cumulative_explained_variance=3)

    @pytest.mark.parametrize("selector", [0, -1, 0.0, 1.5, True, "all"])
    def test_rejects_out_of_range_selector(
        self, frame: pl.DataFrame, selector: object
    ) -> None:
        """Raise for component selectors outside the documented range."""
        model = _fit(frame, "none")

        with pytest.raises(ValueError, match="cumulative_explained_variance"):
            model.reconstruct(_scores(frame, model), selector)  # type: ignore[arg-type]

    def test_rejects_inconsistent_pca_state(self, frame: pl.DataFrame) -> None:
        """Raise when the PCA mean disagrees with the feature columns."""
        model = _fit(frame, "none")
        scores = _scores(frame, model)
        model.pca.mean_ = model.pca.mean_[:-1]

        with pytest.raises(ValueError, match="inconsistent PCA state"):
            model.reconstruct(scores)

    def test_rejects_scaling_model_without_column(self, frame: pl.DataFrame) -> None:
        """Raise when the scaling model lacks a feature column."""
        model = _fit(frame, "z-score")
        scores = _scores(frame, model)
        broken = replace(
            model,
            scaling_model=ScalingModel(
                strategy="z-score",
                centers=dict(model.scaling_model.centers),
                scales={},
            ),
        )

        with pytest.raises(ValueError, match="scaling model"):
            broken.reconstruct(scores)

    def test_rejects_lazy_frame(self, frame: pl.DataFrame) -> None:
        """Accept only an eager ``pl.DataFrame``."""
        model = _fit(frame, "none")

        with pytest.raises(TypeError, match="DataFrame"):
            model.reconstruct(_scores(frame, model).lazy())  # type: ignore[arg-type]


def test_reconstruct_standardized_rejects_out_of_range_component_count(
    frame: pl.DataFrame,
) -> None:
    """Raise when more components are requested than scores provide."""
    model = _fit(frame, "none")
    score_matrix = _scores(frame, model).select("pca-1", "pca-2").to_numpy()

    with pytest.raises(ValueError, match="used_components"):
        reconstruct_standardized(score_matrix, model.pca, 3)
