"""Transform screen: the model, the transform targets, and the shown transform run."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from ..datatable import parse_state
from ..jobs.transform_run import build_transform_config
from ..services.catalog_query import list_files
from ..services.file_table import file_table_config
from ..services.file_table_cache import file_table_context
from ..services.model_settings import model_settings
from ..services.run_dirs import fit_run_reference
from ..services.runs import get_run, latest_run, list_runs, run_status
from ..services.selection import parse_stems_json
from ..services.transform_targets import (
    find_transform_run,
    run_target_files,
    run_target_stems,
)
from ..templating import templates
from ..workspace import TRANSFORM_JOB, Workspace
from .dependencies import get_workspace
from .view_page import render_view_page
from .view_selection import (
    current_model_run,
    current_view_choice,
    model_runs,
    selection_changed,
    selection_changed_trigger,
    view_runs,
)

router = APIRouter(prefix="/transform")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _parse_stems(stems: str) -> list[str]:
    """Parse the chosen stems sent by the transform screen.

    Parameters
    ----------
    stems : str
        JSON array of stems (see ``parse_stems_json``).

    Returns
    -------
    list[str]
        The stems.

    Raises
    ------
    HTTPException
        With status 400 if ``stems`` is not a JSON array of strings.
    """
    try:
        return parse_stems_json(stems)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _submit_message(
    request: Request, context: dict[str, object], trigger: str | None = None
) -> HTMLResponse:
    """Render the message next to the "Run transform" button.

    Parameters
    ----------
    request : Request
        Current request.
    context : dict[str, object]
        Context of ``partials/transform_submit.html``.
    trigger : str | None, default None
        ``HX-Trigger`` header of the response, if any.

    Returns
    -------
    HTMLResponse
        Submit message partial.
    """
    response = templates.TemplateResponse(request, "partials/transform_submit.html", context)
    if trigger is not None:
        response.headers["HX-Trigger"] = trigger
    return response


def _fit_targets(model_run: dict[str, object]) -> list[str]:
    """Return the stems of the model's fit targets for the page.

    Parameters
    ----------
    model_run : dict[str, object]
        The chosen fit run.

    Returns
    -------
    list[str]
        The fit targets, or none when the saved configuration cannot be
        read; the settings summary then reports the error
        (see ``model_settings``).
    """
    try:
        return run_target_stems(model_run)
    except (KeyError, TypeError, ValueError):
        return []


@router.get("", response_class=HTMLResponse)
def transform_page(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the transform page.

    The page chooses the model (a succeeded fit run) and the transform
    targets. With ``use same data for fit`` checked, the targets are the
    files the model was fitted on, shown chosen in the locked table;
    unchecked, the user chooses them from the catalog. Nothing is
    transformed until "Run transform" is pressed.

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
        (see ``render_view_page``), with the first page of the file table in
        its default state (``file_table_context``).
    """
    model_run = current_model_run(workspace)
    model_run_id = None if model_run is None else str(model_run["run_id"])
    use_same_data = workspace.transform_settings.use_same_data
    fit_targets = [] if model_run is None else _fit_targets(model_run)
    config = file_table_config(workspace.settings.metadata_columns)
    return render_view_page(
        request,
        "pages/transform.html",
        {
            "model_runs": model_runs(workspace, model_run_id),
            "model_run_id": model_run_id,
            "model_settings": None if model_run is None else model_settings(model_run),
            "use_same_data": use_same_data,
            "fit_targets": fit_targets,
            "selected": fit_targets if use_same_data else workspace.transform_selection.stems,
            "file_table": config,
            **file_table_context(
                workspace.database,
                config,
                parse_state({}, config),
                workspace.settings.metadata is not None,
            ),
            "status": run_status(latest_run(workspace.database, TRANSFORM_JOB)),
        },
    )


@router.post("/settings")
def update_settings(
    workspace: WorkspaceDependency,
    model: Annotated[str, Form()],
    use_same_data: Annotated[bool, Form()] = False,
    stems: Annotated[str, Form()] = "[]",
) -> Response:
    """Choose the model and whether the transform targets are its fit targets.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    model : str
        Succeeded fit run whose model transforms the targets.
    use_same_data : bool, default False
        Whether the transform targets are the model's fit targets (the
        ``use same data for fit`` checkbox).
    stems : str, default "[]"
        JSON array of the stems chosen in the table. While the checkbox was
        unchecked, they are the user's own choice and are saved (as
        ``POST /transform`` does), so they survive the change; otherwise
        they are the locked fit targets and are ignored.

    Returns
    -------
    Response
        Empty response announcing the change (see ``selection_changed``).

    Raises
    ------
    HTTPException
        With status 400 if ``model`` is not a succeeded fit run or ``stems``
        is not a JSON array of strings.
    """
    requested = _parse_stems(stems)
    if all(run["run_id"] != model for run in model_runs(workspace, model)):
        raise HTTPException(status_code=400, detail=f"succeeded fit run not found: {model}")
    if not workspace.transform_settings.update(model, use_same_data):
        workspace.transform_selection.replace(workspace.database, requested)
    return selection_changed("model")


@router.post("", response_class=HTMLResponse)
def submit_transform(
    request: Request,
    workspace: WorkspaceDependency,
    model: Annotated[str | None, Form()] = None,
    use_same_data: Annotated[bool, Form()] = False,
    stems: Annotated[str, Form()] = "[]",
) -> HTMLResponse:
    """Transform the targets with the page's model, or show an earlier result.

    The model and the checkbox are those the page was drawn with, sent with
    the request, so a change in another tab does not change what the page
    transforms. With ``use same data for fit`` checked, the targets are the
    model's fit targets and ``stems`` is ignored. Unchecked, the chosen
    stems are saved and the cataloged files among them are the targets. If
    a succeeded transform run already transformed the same target files,
    unchanged since, with the same model (see ``find_transform_run``), no
    job is queued and that run is shown instead.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    model : str | None, default None
        Fit run chosen on the page; ``None`` when the page had none.
    use_same_data : bool, default False
        Whether the page's ``use same data for fit`` is checked.
    stems : str, default "[]"
        JSON array of the chosen target stems over all pages and filters,
        parsed by ``parse_stems_json``; stems not in the catalog are ignored.

    Returns
    -------
    HTMLResponse
        Submit message partial: an error when the model is missing or gone,
        or no target is chosen. A reused run is chosen as the shown run and
        announced with ``view-selection-changed`` (``{"changed": "run"}``).
        A queued run is chosen as the shown run, so it is shown once it
        succeeds; the response also replaces the run status out of band and
        triggers ``transform-started``.

    Raises
    ------
    HTTPException
        With status 400 if ``stems`` is not a JSON array of strings.
    """
    requested = _parse_stems(stems)
    if model is None:
        return _submit_message(
            request, {"error": "No succeeded fit run. Run a fit on Preprocess / PCA first."}
        )
    model_run = next(
        (run for run in model_runs(workspace, model) if run["run_id"] == model), None
    )
    if model_run is None:
        return _submit_message(
            request, {"error": f"Fit run {model} is no longer available. Reload the page."}
        )
    if use_same_data:
        files = run_target_files(model_run)
    else:
        wanted = set(workspace.transform_selection.replace(workspace.database, requested))
        files = [
            file for file in list_files(workspace.database) if file["stem"] in wanted
        ]
    if not files:
        return _submit_message(request, {"error": "Choose the files to transform."})
    config = build_transform_config(workspace.settings, files, model_run)
    reused = find_transform_run(
        list_runs(workspace.database, TRANSFORM_JOB, limit=None, status="succeeded"), config
    )
    if reused is not None:
        workspace.view_selection.choose_run(str(reused["run_id"]))
        return _submit_message(request, {"reused": reused}, selection_changed_trigger("run"))
    run_id = workspace.submit_transform(config)
    workspace.view_selection.choose_run(run_id)
    return _submit_message(
        request,
        {"queued": run_status(get_run(workspace.database, run_id))},
        "transform-started",
    )


@router.post("/show")
def show_run(workspace: WorkspaceDependency, run: Annotated[str, Form()]) -> Response:
    """Choose the transform run shown by the display screens.

    The shown files are cleared if the run changes.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : str
        Succeeded transform run.

    Returns
    -------
    Response
        Empty response announcing the change (see ``selection_changed``).

    Raises
    ------
    HTTPException
        With status 400 if ``run`` is not a succeeded transform run.
    """
    if all(candidate["run_id"] != run for candidate in view_runs(workspace, run)):
        raise HTTPException(status_code=400, detail=f"succeeded transform run not found: {run}")
    workspace.view_selection.choose_run(run)
    return selection_changed("run")


@router.get("/runs", response_class=HTMLResponse)
def transform_runs(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the list of transform runs with the shown one.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Run list partial, newest first, with each run's fit run, and the
        shown run with its number of transform targets.
    """
    rows = [
        (run, reference[0] if (reference := fit_run_reference(run)) else None)
        for run in list_runs(workspace.database, TRANSFORM_JOB)
    ]
    return templates.TemplateResponse(
        request,
        "partials/transform_runs.html",
        {"rows": rows, "choice": current_view_choice(workspace)},
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
