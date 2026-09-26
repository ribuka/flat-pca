"""NumPy-side removal of rows with missing feature values."""

from __future__ import annotations

import numpy as np
import polars as pl

MASK_CHUNK_COLUMNS = 8192
"""Column count scanned per step when building the complete-row mask."""


def complete_row_mask(
    values: np.ndarray,
    chunk_columns: int = MASK_CHUNK_COLUMNS,
) -> np.ndarray:
    """Flag the rows of a feature matrix that contain no NaN.

    The matrix is scanned in column chunks so that a full
    ``rows x columns`` boolean matrix is never allocated at once.

    Parameters
    ----------
    values : np.ndarray
        Two-dimensional feature matrix. Nulls converted from Polars appear
        as NaN.
    chunk_columns : int, default MASK_CHUNK_COLUMNS
        Positive number of columns inspected per step.

    Returns
    -------
    np.ndarray
        Boolean vector of length ``values.shape[0]`` that is ``True`` for
        rows without any NaN.

    Raises
    ------
    ValueError
        If ``chunk_columns`` is not positive.
    """
    if chunk_columns <= 0:
        raise ValueError("chunk_columns must be greater than 0")
    mask = np.ones(values.shape[0], dtype=bool)
    for start in range(0, values.shape[1], chunk_columns):
        mask &= ~np.isnan(values[:, start : start + chunk_columns]).any(axis=1)
    return mask


def collect_complete_rows(
    df: pl.LazyFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect the feature matrix and drop rows with any null or NaN.

    Equivalent to filtering ``df`` on every column being non-null and
    non-NaN before collecting, but avoids building one Polars predicate
    per feature column, whose cost grows with the column count.

    Parameters
    ----------
    df : pl.LazyFrame
        Input data containing the selected columns.
    columns : list[str]
        Numeric feature columns to collect.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix of the complete rows, and the boolean mask over the
        input rows that selected them.
    """
    values = df.select(columns).collect().to_numpy()
    mask = complete_row_mask(values)
    if bool(mask.all()):
        return values, mask
    # Boolean indexing yields C order, whereas Polars hands out F order;
    # keep Polars' layout so PCA sees the same memory layout, and hence the
    # same rounding, as a matrix collected from a filtered frame.
    order = "F" if values.flags.f_contiguous else "C"
    return np.asarray(values[mask], order=order), mask


def select_rows(df: pl.LazyFrame, mask: np.ndarray) -> pl.LazyFrame:
    """Keep the rows of a frame flagged by a boolean mask, in order.

    Parameters
    ----------
    df : pl.LazyFrame
        Frame whose row count equals ``len(mask)``.
    mask : np.ndarray
        Boolean vector that is ``True`` for rows to keep.

    Returns
    -------
    pl.LazyFrame
        ``df`` itself when every row is kept, otherwise ``df`` filtered on
        its row position so that the predicate touches a single column.
    """
    if bool(mask.all()):
        return df
    return (
        df.with_row_index("__row_id")
        .filter(pl.col("__row_id").is_in(np.flatnonzero(mask).tolist()))
        .drop("__row_id")
    )
