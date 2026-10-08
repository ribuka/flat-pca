"""Rendering of the screens that show the sidebar's run and files."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import Request
from fastapi.responses import HTMLResponse

from ..templating import templates


def wants_main_only(request: Request) -> bool:
    """Return whether a request asks for the main part of a screen only.

    Parameters
    ----------
    request : Request
        Current request.

    Returns
    -------
    bool
        ``True`` for an htmx request (``HX-Request: true``), which the
        browser sends to replace the main part after the sidebar's choice
        changes.
    """
    return request.headers.get("HX-Request") == "true"


def render_view_page(
    request: Request, name: str, context: Mapping[str, object]
) -> HTMLResponse:
    """Render a screen, or only its main part for an htmx request.

    Parameters
    ----------
    request : Request
        Current request.
    name : str
        Page template extending ``base.html``.
    context : Mapping[str, object]
        Template context of the page.

    Returns
    -------
    HTMLResponse
        Full page, or the contents of ``<main class="content">`` when
        ``wants_main_only`` holds. Both vary with ``HX-Request``, so a cache
        never answers one with the other.
    """
    response = templates.TemplateResponse(
        request, name, {**context, "main_only": wants_main_only(request)}
    )
    response.headers["Vary"] = "HX-Request"
    return response
