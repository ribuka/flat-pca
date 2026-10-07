"""Parts of the page layout shared by every screen."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..services.runs import latest_run_status
from ..services.system_status import read_memory_usage
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


@router.get("/sidebar/system", response_class=HTMLResponse)
def sidebar_system(
    request: Request, workspace: Annotated[Workspace, Depends(get_workspace)]
) -> HTMLResponse:
    """Render the sidebar's memory usage, polled at the configured interval.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Sidebar memory usage partial.
    """
    return templates.TemplateResponse(
        request,
        "partials/sidebar_system.html",
        {
            "memory": read_memory_usage(),
            "poll_seconds": workspace.settings.ui.memory_poll_seconds,
        },
    )
