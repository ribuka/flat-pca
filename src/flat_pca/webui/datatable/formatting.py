"""Formatting table cell values for display."""

from __future__ import annotations

from datetime import datetime


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
