"""Jinja2 template environment of the Web UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


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


templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters["cell"] = format_value
