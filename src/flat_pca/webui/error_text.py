"""Shorten error messages for display in the Web UI."""

from __future__ import annotations

ERROR_TEXT_LIMIT = 200
ELLIPSIS = "…"


def truncate_error(message: object, limit: int = ERROR_TEXT_LIMIT) -> str:
    """Cut an error message down to ``limit`` characters for display.

    Errors such as a metadata CSV listing thousands of repeated keys would
    otherwise fill the screen. Only the displayed text is cut; run records
    and logs keep the full message.

    Parameters
    ----------
    message : object
        Error message, converted with ``str``.
    limit : int, default ERROR_TEXT_LIMIT
        Number of characters kept before the ellipsis.

    Returns
    -------
    str
        ``message`` unchanged when it has at most ``limit`` characters,
        otherwise its first ``limit`` characters followed by ``…``.
    """
    text = str(message)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + ELLIPSIS
