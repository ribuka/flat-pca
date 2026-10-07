"""Check that Q contributions computed from saved artifacts add up to the saved Q.

The Q contribution view computes from ``X.npy`` and ``components.npy``,
saved in ``jobs.artifact_dtype``, while ``scores.parquet`` holds the Q
computed before rounding. Rounding to ``float32`` can lose the variation of
data with a large baseline, so the two may disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .fit_artifacts import DisplayArtifacts, artifact_error
from .scored_samples import scored_samples

# Relative difference above which the contributions are reported as
# disagreeing with the saved Q.
Q_MISMATCH_TOLERANCE = 0.01


@dataclass(frozen=True)
class QMismatch:
    """A file whose Q contributions do not add up to its saved Q.

    Attributes
    ----------
    stem : str
        The file's stem.
    contribution_total : float
        Sum of the unbinned Q contributions of every feature.
    saved_q : float
        The file's Q in ``scores.parquet``.
    """

    stem: str
    contribution_total: float
    saved_q: float

    @property
    def relative_difference(self) -> float:
        """Return ``|contribution_total - saved_q| / |saved_q|``.

        Returns
        -------
        float
            Relative difference, ``inf`` for a saved Q of zero.
        """
        difference = abs(self.contribution_total - self.saved_q)
        return difference / abs(self.saved_q) if self.saved_q else float("inf")


def saved_q_by_stem(artifacts: DisplayArtifacts, spe_column: str) -> dict[str, float]:
    """Return the saved Q of every scored file.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    spe_column : str
        Q column of ``scores.parquet`` (``SpeConfig.spe_column``).

    Returns
    -------
    dict[str, float]
        Q keyed by stem; files dropped by the imputation are absent.

    Raises
    ------
    RunArtifactError
        If ``scores.parquet`` lacks the Q column.
    """
    if spe_column not in artifacts.scores.columns:
        raise artifact_error(ValueError(f"scores.parquet lacks columns {[spe_column]}"))
    samples, scores = scored_samples(artifacts)
    return dict(
        zip(
            samples["stem"].to_list(),
            scores[spe_column].cast(pl.Float64).to_list(),
            strict=True,
        )
    )


def q_mismatch(
    stem: str,
    contribution_total: float,
    saved_q: float,
    tolerance: float = Q_MISMATCH_TOLERANCE,
) -> QMismatch | None:
    """Compare the sum of a file's Q contributions with its saved Q.

    Parameters
    ----------
    stem : str
        The file's stem.
    contribution_total : float
        Sum of the unbinned Q contributions of every feature.
    saved_q : float
        The file's Q in ``scores.parquet``.
    tolerance : float, default Q_MISMATCH_TOLERANCE
        Largest relative difference ``|total - saved| / |saved|`` accepted.

    Returns
    -------
    QMismatch | None
        The disagreement, or ``None`` when the two agree.
    """
    if abs(contribution_total - saved_q) <= tolerance * abs(saved_q):
        return None
    return QMismatch(stem=stem, contribution_total=contribution_total, saved_q=saved_q)
