"""Jinja2 template environment of the Web UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotly
from fastapi.templating import Jinja2Templates

from .services.system_status import app_version, format_gib

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


def static_version(path: str) -> str:
    """Return a version of a static file for its URL's query string.

    Static URLs carry ``?v={static_version(path)}``, so a browser that kept
    an older copy of a changed file (``app.js``, for example) loads the new
    one instead of running the old code against the new pages.

    Parameters
    ----------
    path : str
        Path of the file under ``STATIC_DIR``.

    Returns
    -------
    str
        The file's modification time in nanoseconds.
    """
    return str((STATIC_DIR / path).stat().st_mtime_ns)


@dataclass(frozen=True)
class NavItem:
    """One screen in the sidebar navigation.

    Attributes
    ----------
    label : str
        Text shown in the sidebar.
    path : str | None
        URL path of the screen, or ``None`` while it is not implemented.
    """

    label: str
    path: str | None


NAV_ITEMS = (
    NavItem("Data selection", "/"),
    NavItem("Preprocess / PCA", "/fit"),
    NavItem("transform", "/transform"),
    NavItem("Model", "/model"),
    NavItem("Spectral explorer", "/explore"),
    NavItem("Scores", "/scores"),
    NavItem("T² / Q", "/monitoring"),
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters["cell"] = format_value
templates.env.filters["gib"] = format_gib
templates.env.globals["nav_items"] = NAV_ITEMS
templates.env.globals["app_version"] = app_version()
templates.env.globals["static_version"] = static_version
# plotly.min.js is served from the plotly package (see ``app.plotly_js``), so
# its URL carries the package version instead of a ``static_version``.
templates.env.globals["plotly_version"] = plotly.__version__
