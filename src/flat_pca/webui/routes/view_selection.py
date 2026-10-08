"""The display screens' choices: the model, the transform run, and the shown files."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..services.runs import list_succeeded_runs
from ..services.view_selection import ViewChoice, resolve_view_choice
from ..templating import templates
from ..workspace import FIT_JOB, TRANSFORM_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/sidebar/selection")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def view_runs(workspace: Workspace, requested: str | None) -> list[dict[str, object]]:
    """Return the succeeded transform runs the display screens show.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    requested : str | None
        Explicitly requested run, listed even when it is older than the
        listed runs (see ``list_succeeded_runs``).

    Returns
    -------
    list[dict[str, object]]
        Succeeded transform runs, newest first.
    """
    return list_succeeded_runs(workspace.database, TRANSFORM_JOB, requested)


def model_runs(workspace: Workspace, requested: str | None) -> list[dict[str, object]]:
    """Return the succeeded fit runs whose models can be chosen.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    requested : str | None
        Explicitly requested run, listed even when it is older than the
        listed runs (see ``list_succeeded_runs``).

    Returns
    -------
    list[dict[str, object]]
        Succeeded fit runs, newest first.
    """
    return list_succeeded_runs(workspace.database, FIT_JOB, requested)


def current_model_run(workspace: Workspace) -> dict[str, object] | None:
    """Resolve the fit run whose model the transform screen uses.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.

    Returns
    -------
    dict[str, object] | None
        The chosen fit run, or the latest succeeded fit run when none is
        chosen or the chosen one is gone; ``None`` without succeeded fit
        runs.
    """
    run_id = workspace.transform_settings.model_run_id
    runs = model_runs(workspace, run_id)
    return next((run for run in runs if run["run_id"] == run_id), runs[0] if runs else None)


def current_view_choice(workspace: Workspace) -> ViewChoice:
    """Resolve the shown transform run and the shown files.

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


def selection_changed_trigger(changed: Literal["run", "files", "model"]) -> str:
    """Return the ``HX-Trigger`` header announcing a change of the display screens' choice.

    Parameters
    ----------
    changed : {"run", "files", "model"}
        Choice that was submitted: the shown transform run, the shown
        files, or the transform screen's model and targets. The browser
        refreshes the sidebar's choices and, on a screen that depends on
        this choice, the main part of the screen.

    Returns
    -------
    str
        Header value triggering ``view-selection-changed`` with
        ``{"changed": changed}`` as its detail.
    """
    return json.dumps({"view-selection-changed": {"changed": changed}})


def selection_changed(changed: Literal["run", "files", "model"]) -> Response:
    """Return an empty response announcing a change of the display screens' choice.

    Parameters
    ----------
    changed : {"run", "files", "model"}
        Choice that was submitted (see ``selection_changed_trigger``).

    Returns
    -------
    Response
        Response triggering ``view-selection-changed``.
    """
    return Response(headers={"HX-Trigger": selection_changed_trigger(changed)})


@router.get("", response_class=HTMLResponse)
def view_selection(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the sidebar's shown run and file choices.

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
        since (in another tab, for example), the choice is left unchanged,
        and the change of the run is announced instead, so the browser
        shows the run in use in the sidebar and the main part.
    file : list[str] | None, default None
        Checked stems. Stems that are not transform targets of the run in
        use are ignored, and only the first ``ui.explore_max_files`` of the
        rest (in option order) are kept.

    Returns
    -------
    Response
        Empty response announcing the change (see ``selection_changed``).
    """
    selection = workspace.view_selection
    # Resolving the run in use and replacing its files is one step.
    with selection.transaction():
        choice = current_view_choice(workspace)
        if run != choice.run_id:
            return selection_changed("run")
        wanted = set(file or ())
        stems = [stem for stem in choice.file_options if stem in wanted]
        selection.replace_stems(stems[: workspace.settings.ui.explore_max_files])
    return selection_changed("files")


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
        Transform run of the clicked figure. A figure drawn before the shown
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
                    "message": f"Did not add {stem} from the figure of run {run}, because the shown run "
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
