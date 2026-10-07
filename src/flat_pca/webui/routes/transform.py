"""Transform screen: the data the fitted model transforms for display."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..templating import templates
from ..workspace import Workspace
from .dependencies import get_workspace
from .view_selection import current_view_choice

router = APIRouter(prefix="/transform")


@router.get("", response_class=HTMLResponse)
def transform_page(
    request: Request, workspace: Annotated[Workspace, Depends(get_workspace)]
) -> HTMLResponse:
    """Render the transform page.

    The transform targets are always the files the run was fitted on, so
    the page shows a fixed ``use same data for fit`` checkbox and the
    targets of the fit run chosen in the sidebar.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Full page.
    """
    return templates.TemplateResponse(
        request, "pages/transform.html", {"choice": current_view_choice(workspace)}
    )
