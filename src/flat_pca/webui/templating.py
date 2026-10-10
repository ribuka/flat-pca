"""Jinja2 template environment of the Web UI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import plotly
from fastapi.templating import Jinja2Templates

from .datatable import STATIC_DIR as DATATABLE_STATIC_DIR
from .datatable import configure_environment, format_value
from .error_text import truncate_error
from .services.system_status import app_version, format_gib

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def static_version(path: str, directory: Path = STATIC_DIR) -> str:
    """Return a version of a static file for its URL's query string.

    Static URLs carry ``?v={static_version(path)}``, so a browser that kept
    an older copy of a changed file (``app.js``, for example) loads the new
    one instead of running the old code against the new pages.

    Parameters
    ----------
    path : str
        Path of the file under ``directory``.
    directory : Path, default STATIC_DIR
        Directory of the served files.

    Returns
    -------
    str
        The file's modification time in nanoseconds.
    """
    return str((directory / path).stat().st_mtime_ns)


def script_json(text: str) -> str:
    """Make JSON text safe to embed in a ``<script>`` element.

    Parameters
    ----------
    text : str
        JSON text.

    Returns
    -------
    str
        ``text`` with ``</`` escaped so it cannot close the element.
    """
    return text.replace("</", "<\\/")


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
templates.env.filters["error_text"] = truncate_error
templates.env.globals["nav_items"] = NAV_ITEMS
templates.env.globals["app_version"] = app_version()
templates.env.globals["static_version"] = static_version
templates.env.globals["datatable_static_version"] = partial(
    static_version, directory=DATATABLE_STATIC_DIR
)
configure_environment(templates.env)
# plotly.min.js is served from the plotly package (see ``app.plotly_js``), so
# its URL carries the package version instead of a ``static_version``.
templates.env.globals["plotly_version"] = plotly.__version__
