"""Jinja2 template environment of the Web UI."""

from __future__ import annotations

from dataclasses import dataclass
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
    NavItem("データ選択", "/"),
    NavItem("前処理・PCA", "/fit"),
    NavItem("transform", "/transform"),
    NavItem("モデル", "/model"),
    NavItem("スペクトル探索", "/explore"),
    NavItem("スコア", "/scores"),
    NavItem("T² / Q", "/monitoring"),
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters["cell"] = format_value
templates.env.globals["nav_items"] = NAV_ITEMS
