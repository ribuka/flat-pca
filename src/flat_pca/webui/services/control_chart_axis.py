"""Horizontal axis of the T² and Q control charts."""

from __future__ import annotations

import numpy as np
import polars as pl

from flat_pca.utils import natural_keys

_ROW = "row"
_STEM_RANK = "stem_rank"
_ORDER_VALUE = "order_value"


def control_chart_order(samples: pl.DataFrame, order_by: str | None) -> np.ndarray:
    """Return the row order of the files in the control charts.

    Parameters
    ----------
    samples : pl.DataFrame
        Frame with ``stem`` and the metadata columns.
    order_by : str | None
        Metadata column sorted in ascending order with missing values last;
        ties and ``None`` fall back to the natural order of the stems.

    Returns
    -------
    np.ndarray
        Zero-based row positions of ``samples`` in display order.
    """
    stems = samples["stem"].to_list()
    by_stem = sorted(range(len(stems)), key=lambda row: natural_keys(str(stems[row])))
    ranks = np.empty(len(stems), dtype=np.int64)
    ranks[by_stem] = np.arange(len(stems))
    # Only internal names, so no metadata column name can collide with them.
    frame = pl.DataFrame({_ROW: np.arange(len(stems)), _STEM_RANK: ranks})
    keys = [_STEM_RANK]
    if order_by is not None:
        frame = frame.with_columns(samples[order_by].alias(_ORDER_VALUE))
        keys = [_ORDER_VALUE, _STEM_RANK]
    return frame.sort(keys, nulls_last=True)[_ROW].to_numpy().astype(np.intp)


def supports_numeric_axis(dtype: pl.DataType) -> bool:
    """Return whether a column of ``dtype`` can be drawn on a numeric axis.

    Parameters
    ----------
    dtype : pl.DataType
        Data type of the column.

    Returns
    -------
    bool
        ``True`` for numbers, dates, and datetimes (the last two on a date
        axis); ``False`` for any other type, which is drawn by rank.
    """
    return dtype.is_numeric() or dtype in (pl.Date, pl.Datetime)


def missing_axis_values(values: pl.Series) -> np.ndarray:
    """Return which values cannot be placed on a numeric axis.

    Parameters
    ----------
    values : pl.Series
        Values of a column that supports a numeric axis.

    Returns
    -------
    np.ndarray
        Boolean mask of the null values and, for a float column, the NaN
        values.
    """
    missing = values.is_null()
    if values.dtype.is_float():
        missing = missing | values.is_nan().fill_null(False)
    return missing.to_numpy()
