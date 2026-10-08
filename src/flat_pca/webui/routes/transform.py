"""Transform screen: the data the fitted model transforms for display."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..jobs.transform_run import build_transform_config
from ..services.catalog_query import FileQuery, list_files
from ..services.run_dirs import fit_run_reference
from ..services.runs import get_run, latest_run, list_runs, run_status
from ..services.selection import parse_stems_json
from ..templating import templates
from ..workspace import TRANSFORM_JOB, Workspace
from .dependencies import get_workspace
from .view_page import render_view_page
from .view_selection import current_view_choice

router = APIRouter(prefix="/transform")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


@router.get("", response_class=HTMLResponse)
def transform_page(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the transform page.

    With ``use same data for fit`` checked, the display screens show the
    files the sidebar's fit run was fitted on, and nothing is executed.
    Unchecked, the page lets the user choose the transform targets from the
    catalog and transform them with the model of the sidebar's fit run (the
    fit run of a chosen transform run). The checkbox starts unchecked only
    while the sidebar shows a transform run.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Full page, or its main part for an htmx request
        (see ``render_view_page``).
    """
    choice = current_view_choice(workspace)
    return render_view_page(
        request,
        "pages/transform.html",
        {
            "choice": choice,
            "selected": workspace.transform_selection.stems,
            "status": run_status(latest_run(workspace.database, TRANSFORM_JOB)),
        },
    )


@router.post("", response_class=HTMLResponse)
def submit_transform(
    request: Request,
    workspace: WorkspaceDependency,
    stems: Annotated[str, Form()] = "[]",
) -> HTMLResponse:
    """Save the transform targets and queue a transform job.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    stems : str, default "[]"
        JSON array of the chosen target stems over all pages and filters,
        parsed by ``parse_stems_json``; stems not in the catalog are ignored.

    Returns
    -------
    HTMLResponse
        Submit message partial: an error when no fit run succeeded or no
        target is chosen. When a job is queued, the response also replaces
        the run status out of band and triggers ``transform-started``.

    Raises
    ------
    HTTPException
        With status 400 if ``stems`` is not a JSON array of strings.
    """
    try:
        requested = parse_stems_json(stems)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    selected = workspace.transform_selection.replace(workspace.database, requested)
    fit_run_id = current_view_choice(workspace).fit_run_id
    fit_run = None if fit_run_id is None else get_run(workspace.database, fit_run_id)
    if fit_run is None or not selected:
        error = (
            "No succeeded fit run. Run a fit on Preprocess / PCA first."
            if fit_run is None
            else "Choose the files to transform."
        )
        return templates.TemplateResponse(
            request, "partials/transform_submit.html", {"error": error}
        )
    wanted = set(selected)
    files = [
        file for file in list_files(workspace.database, FileQuery()) if file["stem"] in wanted
    ]
    run_id = workspace.submit_transform(
        build_transform_config(workspace.settings, files, fit_run)
    )
    response = templates.TemplateResponse(
        request,
        "partials/transform_submit.html",
        {"queued": run_status(get_run(workspace.database, run_id))},
    )
    response.headers["HX-Trigger"] = "transform-started"
    return response


@router.get("/runs", response_class=HTMLResponse)
def transform_runs(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the list of transform runs.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Run list partial, newest first, with each run's fit run.
    """
    rows = [
        (run, reference[0] if (reference := fit_run_reference(run)) else None)
        for run in list_runs(workspace.database, TRANSFORM_JOB)
    ]
    return templates.TemplateResponse(
        request,
        "partials/transform_runs.html",
        {"rows": rows, "shown_run_id": current_view_choice(workspace).run_id},
    )


@router.get("/runs/{run_id}/status", response_class=HTMLResponse)
def transform_run_status(
    request: Request, run_id: str, workspace: WorkspaceDependency, polling: bool = False
) -> HTMLResponse:
    """Render the status of one transform run, polled while it is active.

    Parameters
    ----------
    request : Request
        Current request.
    run_id : str
        Transform run to show.
    workspace : Workspace
        Application workspace.
    polling : bool, default False
        Whether the request comes from polling an active run. When such a
        poll finds the run finished, the response triggers
        ``transform-updated`` so the run list and the sidebar reload.

    Returns
    -------
    HTMLResponse
        Run status partial.

    Raises
    ------
    HTTPException
        With status 404 if the run is not a transform run.
    """
    run = get_run(workspace.database, run_id)
    if run is None or run["kind"] != TRANSFORM_JOB:
        raise HTTPException(status_code=404, detail=f"transform run not found: {run_id}")
    status = run_status(run)
    response = templates.TemplateResponse(
        request, "partials/transform_run_status.html", {"status": status}
    )
    if polling and not status["active"]:
        response.headers["HX-Trigger"] = "transform-updated"
    return response

