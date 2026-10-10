"""Data selection screen: catalog updates, file list, and file selection."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response

from ..datatable import apply_state, export_file, parse_export, parse_state
from ..services.catalog_query import category_options, metadata_warnings
from ..services.file_table import file_frame, file_table_config
from ..services.runs import latest_run_status
from ..services.selection import parse_stems_json
from ..templating import templates
from ..workspace import CATALOG_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter()
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


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
            "file_table": file_table_config(workspace.settings.metadata_columns),
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
        Current request whose query parameters are parsed by ``parse_state``
        for the table of ``file_table_config``.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        File table fragment with its column filters, the page links, the
        stems of every file that matches the filters (for the header
        checkbox), and metadata warnings. A page past the last one shows the
        last.

    Raises
    ------
    HTTPException
        With status 400 if the query parameters are invalid.
    """
    config = file_table_config(workspace.settings.metadata_columns)
    parameters = {key: request.query_params.getlist(key) for key in request.query_params}
    try:
        state = parse_state(parameters, config)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    warnings = (
        metadata_warnings(workspace.database) if workspace.settings.metadata else None
    )
    view = apply_state(
        file_frame(workspace.database), state, config, category_options(workspace.database)
    )
    return templates.TemplateResponse(
        request,
        "partials/file_table.html",
        {
            "table": config,
            "view": view,
            "warnings": warnings,
            "selected": set(workspace.selection.stems),
        },
    )


@router.post("/catalog/files/export")
async def export_files(request: Request, workspace: WorkspaceDependency) -> Response:
    """Download the filtered or the selected files of the table as a file.

    Parameters
    ----------
    request : Request
        Current request whose form fields are parsed by ``parse_export``
        for the table of ``file_table_config``: the table's query
        parameters, ``files.export_format`` (``csv`` or ``parquet``),
        ``files.export_rows`` (``filtered`` or ``selected``), and ``stems``
        (the selection as a JSON array).
    workspace : Workspace
        Application workspace.

    Returns
    -------
    Response
        The shown columns of every file matching the filters (on all pages)
        or of every selected file, in the table's sort order, as the
        attachment ``catalog_<YYYYmmdd-HHMMSS>.csv`` (UTF-8 with a byte order
        mark) or ``.parquet``.

    Raises
    ------
    HTTPException
        With status 400 if the form fields are invalid.
    """
    config = file_table_config(workspace.settings.metadata_columns)
    form = await request.form()
    parameters = {key: [str(value) for value in form.getlist(key)] for key in form}
    try:
        export = parse_export(parameters, config)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    # Reading the catalog and writing the file block, so they run in a
    # worker thread and other requests are answered meanwhile.
    file = await run_in_threadpool(
        lambda: export_file(file_frame(workspace.database), export, config, datetime.now().astimezone())
    )
    return Response(file.content, media_type=file.media_type, headers=file.headers)


@router.post("/catalog/selection", response_class=HTMLResponse)
def select_files(
    request: Request,
    workspace: WorkspaceDependency,
    stems: Annotated[str, Form()] = "[]",
) -> HTMLResponse:
    """Replace the selected file set.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    stems : str, default "[]"
        JSON array of the selected file stems over all pages and filters,
        parsed by ``parse_stems_json``; stems not in the catalog are ignored.

    Returns
    -------
    HTMLResponse
        Selection summary partial with a success icon. It triggers
        ``selection-updated`` so the sidebar status refreshes.

    Raises
    ------
    HTTPException
        With status 400 if ``stems`` is not a JSON array of strings.
    """
    try:
        requested = parse_stems_json(stems)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    selected = workspace.selection.replace(workspace.database, requested)
    response = templates.TemplateResponse(
        request, "partials/selection_summary.html", {"selected": selected, "saved": True}
    )
    response.headers["HX-Trigger"] = "selection-updated"
    return response
