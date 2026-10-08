"""The sidebar's choice of the run and the files shown by the display screens."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..services.run_dirs import fit_run_reference
from ..services.runs import get_run, list_succeeded_runs
from ..services.view_selection import ViewChoice, order_view_runs, resolve_view_choice
from ..templating import templates
from ..workspace import FIT_JOB, TRANSFORM_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/sidebar/selection")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def view_runs(workspace: Workspace, requested: str | None) -> list[dict[str, object]]:
    """Return the succeeded runs the display screens choose from.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    requested : str | None
        Explicitly requested run, listed even when it is older than the
        listed runs of its kind (see ``list_succeeded_runs``).

    Returns
    -------
    list[dict[str, object]]
        Fit runs, each followed by its transform runs (see
        ``order_view_runs``). The fit run of every listed transform run is
        listed too, even when it is older than the listed fit runs.
    """
    database = workspace.database
    fit_runs = list_succeeded_runs(database, FIT_JOB, requested)
    transform_runs = list_succeeded_runs(database, TRANSFORM_JOB, requested)
    listed = {str(run["run_id"]) for run in fit_runs}
    for run in transform_runs:
        reference = fit_run_reference(run)
        if reference is None or reference[0] in listed:
            continue
        parent = get_run(database, reference[0])
        if parent is not None and parent["kind"] == FIT_JOB and parent["status"] == "succeeded":
            fit_runs.append(parent)
            listed.add(reference[0])
    fit_runs.sort(key=lambda run: (run["created_at"], run["run_id"]), reverse=True)
    return order_view_runs(fit_runs, transform_runs)


def current_view_choice(workspace: Workspace) -> ViewChoice:
    """Resolve the run and the files chosen in the sidebar.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.

    Returns
    -------
    ViewChoice
        Run in use, its transform targets, and the chosen ones.
    """
    selection = workspace.view_selection
    run_id = selection.run_id
    return resolve_view_choice(
        workspace.cache,
        view_runs(workspace, run_id),
        run_id,
        selection.stems,
    )


def _selection_changed(changed: Literal["run", "files"]) -> Response:
    """Return an empty response announcing a change of the sidebar's choice.

    Parameters
    ----------
    changed : {"run", "files"}
        Choice that was submitted. The browser refreshes the sidebar's
        choices and, on a screen that depends on this choice, the main part
        of the screen.

    Returns
    -------
    Response
        Response triggering ``view-selection-changed`` with
        ``{"changed": changed}`` as its detail.
    """
    trigger = {"view-selection-changed": {"changed": changed}}
    return Response(headers={"HX-Trigger": json.dumps(trigger)})


@router.get("", response_class=HTMLResponse)
def view_selection(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the sidebar's run and file choices.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        View selection partial.
    """
    return templates.TemplateResponse(
        request,
        "partials/view_selection.html",
        {
            "choice": current_view_choice(workspace),
            "max_files": workspace.settings.ui.explore_max_files,
        },
    )


@router.post("/run")
def choose_run(workspace: WorkspaceDependency, run: Annotated[str, Form()]) -> Response:
    """Choose the run; the chosen files are cleared if the run changes.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : str
        Succeeded fit run, or succeeded transform run of a listed fit run.

    Returns
    -------
    Response
        Empty response announcing the change (see ``_selection_changed``).

    Raises
    ------
    HTTPException
        With status 400 if ``run`` is not a run to choose from.
    """
    if all(candidate["run_id"] != run for candidate in view_runs(workspace, run)):
        raise HTTPException(status_code=400, detail=f"succeeded run not found: {run}")
    workspace.view_selection.choose_run(run)
    return _selection_changed("run")


@router.post("/files")
def choose_files(
    workspace: WorkspaceDependency,
    run: Annotated[str, Form()],
    file: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Replace the chosen files of the run the sidebar was drawn with.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : str
        Run in use when the sidebar was drawn. If the run in use has changed
        since (in another tab, for example), the choice is left unchanged
        and the refreshed sidebar shows the current one.
    file : list[str] | None, default None
        Checked stems. Stems that are not transform targets of the run in
        use are ignored, and only the first ``ui.explore_max_files`` of the
        rest (in option order) are kept.

    Returns
    -------
    Response
        Empty response announcing the change (see ``_selection_changed``).
    """
    selection = workspace.view_selection
    # Resolving the run in use and replacing its files is one step.
    with selection.transaction():
        choice = current_view_choice(workspace)
        if run == choice.run_id:
            wanted = set(file or ())
            stems = [stem for stem in choice.file_options if stem in wanted]
            selection.replace_stems(stems[: workspace.settings.ui.explore_max_files])
    return _selection_changed("files")


@router.post("/files/add")
def add_file(
    workspace: WorkspaceDependency,
    run: Annotated[str, Form()],
    stem: Annotated[str, Form()],
) -> JSONResponse:
    """Add one file to the chosen files, as a click on a score point does.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : str
        Fit run of the clicked figure. A figure drawn before the sidebar's
        run changed (in another tab, for example) adds nothing.
    stem : str
        Transform target of the run in use.

    Returns
    -------
    JSONResponse
        ``{"added": true}`` if the file is chosen afterwards; otherwise
        ``{"added": false, "message": ...}`` because ``run`` is no longer
        the run in use or the limit of chosen files is reached. A refusal
        is not an HTTP error, so the browser logs no failed request.

    Raises
    ------
    HTTPException
        With status 400 if ``stem`` is not a transform target of the run.
    """
    selection = workspace.view_selection
    max_files = workspace.settings.ui.explore_max_files
    # Checking the run in use and adding to its files is one step.
    with selection.transaction():
        choice = current_view_choice(workspace)
        if run != choice.run_id:
            return JSONResponse(
                {
                    "added": False,
                    "message": f"Did not add {stem} from the figure of run {run}, because the sidebar's run "
                    f"changed to {choice.run_id}. Reload the page.",
                }
            )
        if stem not in choice.file_options:
            raise HTTPException(status_code=400, detail=f"not a transform target: {stem}")
        added = selection.add_stem(stem, choice.file_options, max_files)
    if added:
        return JSONResponse({"added": True})
    return JSONResponse(
        {
            "added": False,
            "message": f"Up to {max_files} shown files can be chosen. "
            f"To add {stem}, clear another file in the sidebar.",
        }
    )
