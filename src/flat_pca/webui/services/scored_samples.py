"""Scored files of a fit run with their metadata."""

from __future__ import annotations

import numpy as np
import polars as pl

from flat_pca.spectral.schema import SOURCE_COLUMN

from .fit_artifacts import DisplayArtifacts

# Columns of samples.parquet that are not metadata.
SAMPLE_KEY_COLUMNS = (SOURCE_COLUMN, "stem")


def metadata_columns(samples: pl.DataFrame) -> list[str]:
    """Return the metadata columns of ``samples.parquet``.

    Parameters
    ----------
    samples : pl.DataFrame
        ``samples.parquet`` of a run.

    Returns
    -------
    list[str]
        Every column but ``SAMPLE_KEY_COLUMNS``, in file order.
    """
    return [column for column in samples.columns if column not in SAMPLE_KEY_COLUMNS]


def choose_metadata_column(
    requested: str | None, default: str | None, options: list[str]
) -> str | None:
    """Return the chosen metadata column.

    Parameters
    ----------
    requested : str | None
        Requested column; ``""`` for none and ``None`` for the default.
    default : str | None
        Default column from the settings.
    options : list[str]
        Metadata columns of the run's samples.

    Returns
    -------
    str | None
        The requested column if available, otherwise the default column if
        available, otherwise ``None``.
    """
    if requested == "":
        return None
    for candidate in (requested, default):
        if candidate in options:
            return candidate
    return None


def scored_samples(artifacts: DisplayArtifacts) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the scored files' metadata and their ``scores.parquet`` rows.

    The two frames are kept apart, so a metadata column may have any name,
    including a score column's.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``stem`` and the metadata columns of each scored file, and the
        matching rows of ``scores.parquet``, both in ``samples.parquet``
        order. Files dropped by the imputation have no score and are absent.
    """
    positions = {
        source: position
        for position, source in enumerate(artifacts.samples[SOURCE_COLUMN].to_list())
    }
    sample_rows = np.array(
        [positions[source] for source in artifacts.scores[SOURCE_COLUMN].to_list()],
        dtype=np.intp,
    )
    by_sample = np.argsort(sample_rows, kind="stable")
    return (
        artifacts.samples[sample_rows[by_sample]].drop(SOURCE_COLUMN),
        artifacts.scores[by_sample],
    )
