"""Data selection screen: catalog updates, file list, and file selection."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..services.catalog_query import (
    category_options,
    list_files,
    metadata_warnings,
    parse_file_query,
)
from ..services.pagination import paginate
from ..services.runs import latest_run_status
from ..templating import templates
from ..workspace import CATALOG_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter()
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]

FILE_PAGE_SIZE = 1000


@router.get("/", response_class=HTMLResponse)
def data_selection_page(
    request: Request, workspace: WorkspaceDependency
) -> HTMLResponse:
    """Render the data selection page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Full page with the catalog status and the file table, which loads
        separately, and the saved selection.
    """
    return templates.TemplateResponse(
        request,
        "pages/data_selection.html",
        {
            **latest_run_status(workspace.database, CATALOG_JOB),
            "selected": workspace.selection.stems,
        },
    )


@router.get("/catalog/status", response_class=HTMLResponse)
def catalog_status(
    request: Request, workspace: WorkspaceDependency, polling: bool = False
) -> HTMLResponse:
    """Render the catalog status partial polled while a catalog run is active.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    polling : bool, default False
        Whether the request comes from polling an active run. When such a
        poll finds the run finished, the response triggers
        ``catalog-updated`` so the file table reloads.

    Returns
    -------
    HTMLResponse
        Status partial.
    """
    context = latest_run_status(workspace.database, CATALOG_JOB)
    response = templates.TemplateResponse(
        request, "partials/catalog_status.html", context
    )
    if polling and not context["active"]:
        response.headers["HX-Trigger"] = "catalog-updated"
    return response


@router.post("/catalog/refresh", response_class=HTMLResponse)
def refresh_catalog(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Queue a catalog update and render its status.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Status partial of the queued (or already active) catalog run. It
        triggers ``catalog-started`` so the sidebar status refreshes.
    """
    workspace.submit_catalog()
    response = templates.TemplateResponse(
        request,
        "partials/catalog_status.html",
        latest_run_status(workspace.database, CATALOG_JOB),
    )
    response.headers["HX-Trigger"] = "catalog-started"
    return response


@router.get("/catalog/files", response_class=HTMLResponse)
def file_table(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render one page of the filtered and sorted file table.

    Parameters
    ----------
    request : Request
        Current request whose query parameters are parsed by
        ``parse_file_query``.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        File table partial with its column filters, the page links, the
        stems of every file that matches the filters (for the header
        checkbox), and metadata warnings. Each page holds
        ``FILE_PAGE_SIZE`` files; a page past the last one shows the last.

    Raises
    ------
    HTTPException
        With status 400 if the query parameters are invalid.
    """
    columns = workspace.settings.metadata_columns
    try:
        query = parse_file_query(request.query_params, columns)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    warnings = (
        metadata_warnings(workspace.database) if workspace.settings.metadata else None
    )
    files = list_files(workspace.database, query)
    return templates.TemplateResponse(
        request,
        "partials/file_table.html",
        {
            "page": paginate(files, query.page, FILE_PAGE_SIZE),
            "matching_stems": [file["stem"] for file in files],
            "columns": columns,
            "options": category_options(workspace.database),
            "query": query,
            "warnings": warnings,
            "selected": set(workspace.selection.stems),
        },
    )


@router.post("/catalog/selection", response_class=HTMLResponse)
def select_files(
    request: Request,
    workspace: WorkspaceDependency,
    stems: Annotated[list[str] | None, Form()] = None,
) -> HTMLResponse:
    """Replace the selected file set.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    stems : list[str] | None, default None
        Selected file stems over all pages and filters; stems not in the
        catalog are ignored.

    Returns
    -------
    HTMLResponse
        Selection summary partial with a success icon. It triggers
        ``selection-updated`` so the sidebar status refreshes.
    """
    selected = workspace.selection.replace(workspace.database, stems or [])
    response = templates.TemplateResponse(
        request, "partials/selection_summary.html", {"selected": selected, "saved": True}
    )
    response.headers["HX-Trigger"] = "selection-updated"
    return response
