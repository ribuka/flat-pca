"""Data selection screen: catalog updates, file list, and file selection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..jobs.progress import read_progress
from ..services.catalog_query import (
    FILE_SORT_COLUMNS,
    category_options,
    list_files,
    metadata_warnings,
    parse_file_query,
)
from ..services.runs import ACTIVE_STATUSES, latest_run
from ..templating import templates
from ..workspace import CATALOG_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter()
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _status_context(workspace: Workspace) -> dict[str, object]:
    """Collect the latest catalog run and its progress.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.

    Returns
    -------
    dict[str, object]
        ``run`` (or ``None``), ``active``, and ``progress`` (or ``None``).
    """
    run = latest_run(workspace.database, CATALOG_JOB)
    active = run is not None and run["status"] in ACTIVE_STATUSES
    progress = read_progress(Path(str(run["artifact_dir"]))) if active else None
    return {"run": run, "active": active, "progress": progress}


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
        Full page with the catalog status, filter form, and file table.
    """
    return templates.TemplateResponse(
        request,
        "pages/data_selection.html",
        {
            **_status_context(workspace),
            "columns": workspace.settings.metadata_columns,
            "sort_columns": [*FILE_SORT_COLUMNS, *workspace.settings.metadata_columns],
            "options": category_options(workspace.database),
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
    context = _status_context(workspace)
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
        Status partial of the queued (or already active) catalog run.
    """
    workspace.submit_catalog()
    return templates.TemplateResponse(
        request, "partials/catalog_status.html", _status_context(workspace)
    )


@router.get("/catalog/files", response_class=HTMLResponse)
def file_table(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the filtered and sorted file table.

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
        File table partial with metadata warnings.

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
    return templates.TemplateResponse(
        request,
        "partials/file_table.html",
        {
            "files": list_files(workspace.database, query),
            "columns": columns,
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
        Checked file stems; stems not in the catalog are ignored.

    Returns
    -------
    HTMLResponse
        Selection summary partial.
    """
    selected = workspace.selection.replace(workspace.database, stems or [])
    return templates.TemplateResponse(
        request, "partials/selection_summary.html", {"selected": selected}
    )
