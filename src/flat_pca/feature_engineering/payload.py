"""Readers for entries of JSON-compatible fitted-state payloads."""

from __future__ import annotations

from typing import cast


def read_float_map(
    payload: dict[str, object],
    key: str,
    *,
    required: bool = False,
) -> dict[str, float]:
    """Read one payload entry as a column-to-float mapping.

    Parameters
    ----------
    payload : dict[str, object]
        Serialized state to read from.
    key : str
        Entry name.
    required : bool, default False
        If ``True``, a missing entry raises ``KeyError``. If ``False``, a
        missing entry is treated as an empty mapping, which keeps payloads
        written before the optional entries existed readable.

    Returns
    -------
    dict[str, float]
        Per-column floating-point values.

    Raises
    ------
    KeyError
        If ``required`` is ``True`` and the entry is missing.
    """
    raw = cast(
        dict[str, float],
        payload[key] if required else payload.get(key, {}),
    )
    return {column: float(value) for column, value in raw.items()}
