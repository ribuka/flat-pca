"""Tests for missing-value imputation."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.pca.impute import (
    DEFAULT_KMEANS_N_CLUSTERS,
    ImputeModel,
    apply_kmeans_impute,
    fit_impute,
    fit_kmeans_impute,
    resolve_kmeans_n_clusters,
)


class TestResolveKmeansNClusters:
    """Tests for ``resolve_kmeans_n_clusters``."""

    def test_uses_default_when_none(self) -> None:
        """Fall back to the module default when no count is requested."""
        assert resolve_kmeans_n_clusters(None, complete_row_count=100) == (
            DEFAULT_KMEANS_N_CLUSTERS
        )

    def test_uses_requested_value_when_within_bounds(self) -> None:
        """Use the requested cluster count when it fits the complete rows."""
        assert resolve_kmeans_n_clusters(3, complete_row_count=100) == 3

    def test_clips_to_complete_row_count(self) -> None:
        """Clip a requested count that exceeds the available complete rows."""
        assert resolve_kmeans_n_clusters(10, complete_row_count=4) == 4
        assert resolve_kmeans_n_clusters(None, complete_row_count=2) == 2

    def test_rejects_zero_complete_rows(self) -> None:
        """Reject fitting when no row is free of missing values."""
        with pytest.raises(ValueError, match="at least one row"):
            resolve_kmeans_n_clusters(None, complete_row_count=0)


class TestFitKmeansImpute:
    """Tests for ``fit_kmeans_impute``."""

    def test_fills_missing_values_from_nearest_centroid(self) -> None:
        """Fill each missing feature value from its row's nearest centroid."""
        frame = pl.DataFrame(
            {
                "a": [0.0, 0.0, 10.0, 10.0, None],
                "b": [0.0, 0.1, 10.0, 9.9, 0.05],
            }
        )
        prepared, n_clusters, centroids = fit_kmeans_impute(
            frame.lazy(), ["a", "b"], n_clusters=2
        )
        result = prepared.collect()

        assert n_clusters == 2
        assert len(centroids) == 2
        assert result.height == 5
        assert result["b"].to_list() == pytest.approx([0.0, 0.1, 10.0, 9.9, 0.05])
        # Row 4 is missing "a" but close to the low cluster in "b"; it should
        # be filled with that cluster's "a" centroid, near 0.
        assert result["a"][4] == pytest.approx(0.0, abs=1.0)

    def test_clips_n_clusters_to_complete_row_count(self) -> None:
        """Fit fewer clusters than requested when few complete rows exist."""
        frame = pl.DataFrame(
            {
                "a": [1.0, 2.0, None],
                "b": [1.0, 2.0, 3.0],
            }
        )
        _, n_clusters, centroids = fit_kmeans_impute(
            frame.lazy(), ["a", "b"], n_clusters=5
        )

        assert n_clusters == 2
        assert len(centroids) == 2

    def test_raises_when_no_complete_rows(self) -> None:
        """Reject fitting when every row has a missing feature value."""
        frame = pl.DataFrame({"a": [None, 1.0], "b": [1.0, None]})

        with pytest.raises(ValueError, match="at least one row"):
            fit_kmeans_impute(frame.lazy(), ["a", "b"], n_clusters=1)

    def test_is_deterministic_across_repeated_fits(self) -> None:
        """Produce identical centroids and fills across repeated fits."""
        frame = pl.DataFrame(
            {
                "a": [0.0, 0.0, 10.0, 10.0, None, None],
                "b": [0.0, 0.1, 10.0, 9.9, 0.05, 10.05],
            }
        )
        first, _, first_centroids = fit_kmeans_impute(
            frame.lazy(), ["a", "b"], n_clusters=2
        )
        second, _, second_centroids = fit_kmeans_impute(
            frame.lazy(), ["a", "b"], n_clusters=2
        )

        assert first.collect().equals(second.collect())
        assert first_centroids == second_centroids

    def test_leaves_complete_rows_unchanged(self) -> None:
        """Never modify a row that already has every feature value."""
        frame = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0],
            }
        )
        prepared, _, _ = fit_kmeans_impute(frame.lazy(), ["a", "b"], n_clusters=2)

        assert prepared.collect().equals(frame)


class TestApplyKmeansImpute:
    """Tests for ``apply_kmeans_impute``."""

    def test_fills_missing_values_using_fitted_centroids(self) -> None:
        """Reuse fitted centroids to fill missing values in new data."""
        centroids = [{"a": 0.0, "b": 0.0}, {"a": 10.0, "b": 10.0}]
        frame = pl.DataFrame({"a": [None, 9.5], "b": [0.2, None]})

        result = apply_kmeans_impute(frame.lazy(), ["a", "b"], centroids).collect()

        assert result["a"][0] == pytest.approx(0.0)
        assert result["b"][1] == pytest.approx(10.0)

    def test_leaves_complete_rows_unchanged(self) -> None:
        """Never modify a row that already has every feature value."""
        centroids = [{"a": 0.0, "b": 0.0}, {"a": 10.0, "b": 10.0}]
        frame = pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})

        result = apply_kmeans_impute(frame.lazy(), ["a", "b"], centroids).collect()

        assert result.equals(frame)

    def test_ignores_missing_dimensions_when_choosing_nearest_centroid(self) -> None:
        """Choose the nearest centroid using only the row's present columns."""
        centroids = [
            {"a": 0.0, "b": 0.0, "c": 100.0},
            {"a": 0.0, "b": 0.0, "c": -100.0},
        ]
        # "c" is missing, and "a"/"b" tie between clusters, so the choice
        # must not be influenced by the missing dimension.
        frame = pl.DataFrame({"a": [0.0], "b": [0.0], "c": [None]})

        result = apply_kmeans_impute(frame.lazy(), ["a", "b", "c"], centroids).collect()

        assert result["c"][0] in (100.0, -100.0)
        assert np.isfinite(result["c"][0])


class TestImputeModel:
    """Tests for ``fit_impute`` and ``ImputeModel``."""

    def test_median_apply_reuses_fitted_values(self) -> None:
        """Fill new data with the medians fitted on the training data."""
        train = pl.DataFrame({"a": [1.0, 2.0, 9.0, None], "b": [4.0, 5.0, 6.0, 7.0]})
        prepared, model = fit_impute(train.lazy(), ["a", "b"], "median")

        result = model.apply(
            pl.DataFrame({"a": [None, float("nan")], "b": [None, 0.0]}).lazy(),
            ["a", "b"],
        ).collect()

        assert model.values == {"a": 2.0, "b": 5.5}
        assert prepared.collect()["a"].to_list() == [1.0, 2.0, 9.0, 2.0]
        assert result.to_dict(as_series=False) == {"a": [2.0, 2.0], "b": [5.5, 0.0]}

    def test_drop_apply_removes_missing_rows(self) -> None:
        """Drop rows with a null or NaN feature value without fitting values."""
        frame = pl.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, float("nan")]})
        prepared, model = fit_impute(frame.lazy(), ["a", "b"], "drop")

        assert model.values == {}
        assert model.kmeans_n_clusters is None
        assert prepared.collect().to_dict(as_series=False) == {"a": [1.0], "b": [1.0]}
        assert model.apply(frame.lazy(), ["a", "b"]).collect().equals(
            prepared.collect()
        )

    def test_kmeans_matches_fit_kmeans_impute(self) -> None:
        """Hold the cluster count and centroids of ``fit_kmeans_impute``."""
        frame = pl.DataFrame(
            {"a": [0.0, 0.2, 10.0, 10.2, None], "b": [0.0, 0.1, 10.0, 9.9, 10.1]}
        )
        expected, n_clusters, centroids = fit_kmeans_impute(
            frame.lazy(), ["a", "b"], n_clusters=2
        )

        prepared, model = fit_impute(frame.lazy(), ["a", "b"], "kmeans", 2)

        assert model.kmeans_n_clusters == n_clusters
        assert model.kmeans_centroids == centroids
        assert prepared.collect().equals(expected.collect())
        assert model.apply(frame.lazy(), ["a", "b"]).collect().equals(
            expected.collect()
        )

    def test_payload_round_trip(self) -> None:
        """Restore the same model from its payload."""
        model = ImputeModel(
            strategy="kmeans",
            values={},
            kmeans_n_clusters=1,
            kmeans_centroids=[{"a": 1.0, "b": 2.0}],
        )

        payload = model.to_payload()

        assert list(payload) == [
            "impute_strategy",
            "impute_values",
            "impute_kmeans_n_clusters",
            "impute_kmeans_centroids",
        ]
        assert ImputeModel.from_payload(payload) == model

    @pytest.mark.parametrize(
        "key",
        [
            "impute_strategy",
            "impute_values",
            "impute_kmeans_n_clusters",
            "impute_kmeans_centroids",
        ],
    )
    def test_from_payload_rejects_missing_entry(self, key: str) -> None:
        """Raise ``KeyError`` when a payload entry is missing."""
        payload = ImputeModel(
            strategy="median",
            values={"a": 1.0},
            kmeans_n_clusters=None,
            kmeans_centroids=[],
        ).to_payload()
        del payload[key]

        with pytest.raises(KeyError, match=key):
            ImputeModel.from_payload(payload)
