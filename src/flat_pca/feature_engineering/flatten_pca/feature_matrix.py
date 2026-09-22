"""NumPy feature-matrix assembly for Flatten-PCA flattening."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from flat_pca.spectral.schema import SOURCE_COLUMN

from ...utils.pl_snippets import validate_missing_ratio_threshold

MetadataKey = tuple[int, int, float]


def build_feature_matrix(
    frames: Sequence[pl.DataFrame],
    wavelength_columns_sorted: list[str],
    metadata_rows: list[MetadataKey],
) -> np.ndarray:
    """Assemble flattened feature values as one dense ``float64`` matrix.

    Each frame contributes one matrix row holding its
    ``(wavelength, Step, Sequence, StepTime)`` values in flatten order:
    ascending wavelength first, then the sorted metadata combinations. The
    per-frame block is built as ``(combination, wavelength)`` and transposed
    before raveling, so a feature's position is
    ``wavelength index * combination count + combination index``, which is
    exactly the order ``flatten_inputs`` produces.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame]
        Validated spectral frames, already ordered by normalized input path.
    wavelength_columns_sorted : list[str]
        Wavelength column names ordered by numeric wavelength.
    metadata_rows : list[MetadataKey]
        Sorted union of ``(Step, Sequence, StepTime)`` combinations across
        ``frames``.

    Returns
    -------
    np.ndarray
        ``(frame count, wavelength count * combination count)`` matrix.
        Combinations a frame does not cover stay ``NaN``.

    Notes
    -----
    A frame carrying duplicate ``(Step, Sequence, StepTime)`` rows scatters
    them into the same block row, and NumPy's fancy-index assignment keeps
    the last write, matching the dictionary-based flattening this replaces.
    """
    row_index = {key: index for index, key in enumerate(metadata_rows)}
    combination_count = len(metadata_rows)
    wavelength_count = len(wavelength_columns_sorted)
    matrix = np.empty(
        (len(frames), wavelength_count * combination_count), dtype=np.float64
    )
    for frame_index, frame in enumerate(frames):
        block = np.full((combination_count, wavelength_count), np.nan, dtype=np.float64)
        positions = np.fromiter(
            (
                row_index[key]
                for key in frame.select("Step", "Sequence", "StepTime").iter_rows()
            ),
            dtype=np.intp,
            count=frame.height,
        )
        values = (
            frame.select(wavelength_columns_sorted)
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        block[positions] = values
        matrix[frame_index] = block.T.ravel()
    return matrix


def select_dense_features(
    matrix: np.ndarray,
    feature_names: list[str],
    max_null_ratio: float,
) -> tuple[np.ndarray, list[str]]:
    """Keep only feature columns whose missing-value ratio is low enough.

    Parameters
    ----------
    matrix : np.ndarray
        Feature matrix from ``build_feature_matrix``, where missing values
        are ``NaN``.
    feature_names : list[str]
        Feature names positionally matching ``matrix`` columns.
    max_null_ratio : float
        Inclusive upper bound (0.0-1.0) on a column's missing-value ratio.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The retained columns and their names, in their original order.

    Raises
    ------
    ValueError
        If ``max_null_ratio`` is not between 0.0 and 1.0.
    """
    validate_missing_ratio_threshold(max_null_ratio)
    if matrix.shape[0] == 0:
        return matrix, feature_names
    keep = np.isnan(matrix).mean(axis=0) <= max_null_ratio
    kept_names = [
        name for name, is_kept in zip(feature_names, keep.tolist(), strict=True) if is_kept
    ]
    return matrix[:, keep], kept_names


def build_flattened_frame(
    matrix: np.ndarray,
    sources: list[str],
    feature_names: list[str],
) -> pl.DataFrame:
    """Wrap a feature matrix as a flattened frame with ``source`` first.

    Parameters
    ----------
    matrix : np.ndarray
        Feature matrix whose rows correspond to ``sources`` and whose
        columns correspond to ``feature_names``.
    sources : list[str]
        Normalized input path texts, one per matrix row.
    feature_names : list[str]
        Feature column names, one per matrix column.

    Returns
    -------
    pl.DataFrame
        ``source`` followed by the feature columns. ``NaN`` entries become
        polars nulls, so missing combinations keep the null representation
        that downstream pruning and imputation expect.
    """
    frame = pl.DataFrame(
        matrix,
        schema=feature_names,
        orient="row",
        nan_to_null=True,
    )
    return frame.insert_column(
        0, pl.Series(SOURCE_COLUMN, sources, dtype=pl.String)
    )

