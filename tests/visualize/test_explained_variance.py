"""Tests for the scree plot."""

import numpy as np
from sklearn.decomposition import PCA

from flat_pca.feature_engineering.pca.analysis import explained_variance_table
from flat_pca.visualize import create_scree_plot


def test_scree_plot_draws_ratios_and_their_running_sum() -> None:
    """Bars show each ratio and the line shows the cumulative ratio."""
    rng = np.random.default_rng(0)
    pca = PCA(n_components=3).fit(rng.normal(size=(20, 4)))
    table = explained_variance_table(pca, ("pca-1", "pca-2", "pca-3"))

    figure = create_scree_plot(table)

    bars, line = figure.data
    assert list(bars.x) == [1, 2, 3]
    np.testing.assert_allclose(bars.y, pca.explained_variance_ratio_)
    np.testing.assert_allclose(line.y, np.cumsum(pca.explained_variance_ratio_))
