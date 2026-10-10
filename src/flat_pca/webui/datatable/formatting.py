"""Formatting table cell values and column types for display."""

from __future__ import annotations

from datetime import datetime

import polars as pl

# Short names of the polars types without parameters, as polars prints them
# over a frame's columns.
SHORT_TYPE_NAMES: dict[type[pl.DataType], str] = {
    pl.String: "str",
    pl.Categorical: "cat",
    pl.Enum: "enum",
    pl.Boolean: "bool",
    pl.Int8: "i8",
    pl.Int16: "i16",
    pl.Int32: "i32",
    pl.Int64: "i64",
    pl.Int128: "i128",
    pl.UInt8: "u8",
    pl.UInt16: "u16",
    pl.UInt32: "u32",
    pl.UInt64: "u64",
    pl.UInt128: "u128",
    pl.Float32: "f32",
    pl.Float64: "f64",
    pl.Date: "date",
    pl.Time: "time",
    pl.Binary: "binary",
    pl.Null: "null",
    pl.Object: "object",
}


def format_value(value: object) -> str:
    """Format a table cell value for display.

    Parameters
    ----------
    value : object
        Cell value.

    Returns
    -------
    str
        Empty text for ``None``, ``YYYY-mm-dd HH:MM:SS`` for datetimes,
        up to six significant digits for floats, and ``str(value)`` otherwise.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"{value:%Y-%m-%d %H:%M:%S}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _time_unit(unit: str) -> str:
    """Return a time unit as polars prints it: ``μs`` for microseconds."""
    return "μs" if unit == "us" else unit


def dtype_label(dtype: pl.DataType) -> str:
    """Return the short name of a column type shown under a column name.

    Parameters
    ----------
    dtype : pl.DataType
        Polars type of the column.

    Returns
    -------
    str
        The name polars prints over a frame's columns, such as ``str``,
        ``cat``, ``i64``, ``f64``, ``datetime[μs]``,
        ``datetime[ns, UTC]``, ``duration[ms]``, ``decimal[10,2]``, or
        ``list[i64]``; ``str(dtype)`` for other types.
    """
    if isinstance(dtype, pl.Datetime):
        zone = f", {dtype.time_zone}" if dtype.time_zone else ""
        return f"datetime[{_time_unit(dtype.time_unit)}{zone}]"
    if isinstance(dtype, pl.Duration):
        return f"duration[{_time_unit(dtype.time_unit)}]"
    if isinstance(dtype, pl.Decimal):
        return f"decimal[{dtype.precision},{dtype.scale}]"
    if isinstance(dtype, pl.List):
        return f"list[{dtype_label(dtype.inner)}]"
    if isinstance(dtype, pl.Array):
        leaf = dtype.inner
        while isinstance(leaf, pl.Array):
            leaf = leaf.inner
        shape = dtype.shape[0] if len(dtype.shape) == 1 else dtype.shape
        return f"array[{dtype_label(leaf)}, {shape}]"
    if isinstance(dtype, pl.Struct):
        return f"struct[{len(dtype.fields)}]"
    return SHORT_TYPE_NAMES.get(dtype.base_type(), str(dtype))
