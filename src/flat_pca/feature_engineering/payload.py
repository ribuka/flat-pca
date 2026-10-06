"""Readers for entries of JSON-compatible fitted-state payloads."""

from __future__ import annotations

from typing import cast


def read_float_map(payload: dict[str, object], key: str) -> dict[str, float]:
    """Read one payload entry as a column-to-float mapping.

    Parameters
    ----------
    payload : dict[str, object]
        Serialized state to read from.
    key : str
        Entry name.

    Returns
    -------
    dict[str, float]
        Per-column floating-point values.

    Raises
    ------
    KeyError
        If the entry is missing.
    """
    raw = cast(dict[str, float], payload[key])
    return {column: float(value) for column, value in raw.items()}
