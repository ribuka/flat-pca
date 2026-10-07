"""Parts of the page layout shared by every screen."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..services.runs import latest_run_status
from ..templating import templates
from ..workspace import CATALOG_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter()


@router.get("/sidebar/status", response_class=HTMLResponse)
def sidebar_status(
    request: Request, workspace: Annotated[Workspace, Depends(get_workspace)]
) -> HTMLResponse:
    """Render the sidebar's catalog and selection status.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Sidebar status partial.
    """
    return templates.TemplateResponse(
        request,
        "partials/sidebar_status.html",
        {
            "catalog": latest_run_status(workspace.database, CATALOG_JOB),
            "selected_count": len(workspace.selection.stems),
        },
    )
