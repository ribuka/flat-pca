"""The sidebar's choice of the fit run and the files shown by the display screens."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..services.runs import list_succeeded_runs
from ..services.view_selection import ViewChoice, resolve_view_choice
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/sidebar/selection")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def current_view_choice(workspace: Workspace) -> ViewChoice:
    """Resolve the fit run and the files chosen in the sidebar.

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
        list_succeeded_runs(workspace.database, FIT_JOB, run_id),
        run_id,
        selection.stems,
    )


def _reload() -> Response:
    """Return an empty response that makes htmx reload the current page.

    Returns
    -------
    Response
        Response with ``HX-Refresh: true``.
    """
    return Response(headers={"HX-Refresh": "true"})


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
    """Choose the fit run; the chosen files are cleared if the run changes.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : str
        Succeeded fit run.

    Returns
    -------
    Response
        Empty response reloading the page.

    Raises
    ------
    HTTPException
        With status 400 if ``run`` is not a succeeded fit run.
    """
    runs = list_succeeded_runs(workspace.database, FIT_JOB, run)
    if all(candidate["run_id"] != run for candidate in runs):
        raise HTTPException(status_code=400, detail=f"succeeded fit run not found: {run}")
    workspace.view_selection.choose_run(run)
    return _reload()


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
        and the reload shows the current one.
    file : list[str] | None, default None
        Checked stems. Stems that are not transform targets of the run in
        use are ignored, and only the first ``ui.explore_max_files`` of the
        rest (in option order) are kept.

    Returns
    -------
    Response
        Empty response reloading the page.
    """
    selection = workspace.view_selection
    # Resolving the run in use and replacing its files is one step.
    with selection.transaction():
        choice = current_view_choice(workspace)
        if run == choice.run_id:
            wanted = set(file or ())
            stems = [stem for stem in choice.file_options if stem in wanted]
            selection.replace_stems(stems[: workspace.settings.ui.explore_max_files])
    return _reload()


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
