"""Tests for the score points of the score screen."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flat_pca.webui.services.fit_artifacts import DisplayArtifacts, RunArtifactError
from flat_pca.webui.services.scores import score_table


def _artifacts(samples: pl.DataFrame) -> DisplayArtifacts:
    """Return artifacts of three files, the second of which has no score.

    Parameters
    ----------
    samples : pl.DataFrame
        ``samples.parquet`` of the files ``a``, ``b``, and ``c``.

    Returns
    -------
    DisplayArtifacts
        Artifacts whose scores list ``c`` before ``a``.
    """
    return DisplayArtifacts(
        features=pl.DataFrame({"feature": ["f"]}),
        samples=samples,
        scores=pl.DataFrame(
            {
                "source": ["c.parquet", "a.parquet"],
                "pca-1": [3.0, 1.0],
                "pca-2": [-3.0, -1.0],
            }
        ),
        x=np.zeros((3, 1)),
        components=np.zeros((2, 1)),
    )


def test_metadata_named_like_scores_or_work_columns_is_kept_apart() -> None:
    """Metadata columns named ``PC1`` or ``__order`` neither replace nor break the scores."""
    samples = pl.DataFrame(
        {
            "source": ["a.parquet", "b.parquet", "c.parquet"],
            "stem": ["a", "b", "c"],
            "PC1": ["x", "y", "z"],
            "__order": [10, 20, 30],
        }
    )

    points = score_table(_artifacts(samples), ("pca-1", "pca-2"), 1, 2)

    assert points.samples.to_dict(as_series=False) == {
        "stem": ["a", "c"],
        "PC1": ["x", "z"],
        "__order": [10, 30],
    }
    np.testing.assert_array_equal(points.x, [1.0, 3.0])
    np.testing.assert_array_equal(points.y, [-1.0, -3.0])


def test_missing_score_columns_are_reported() -> None:
    """A score column absent from ``scores.parquet`` asks to execute the run again."""
    samples = pl.DataFrame(
        {"source": ["a.parquet", "b.parquet", "c.parquet"], "stem": ["a", "b", "c"]}
    )

    with pytest.raises(RunArtifactError, match="pca-3"):
        score_table(_artifacts(samples), ("pca-1", "pca-2", "pca-3"), 1, 3)
