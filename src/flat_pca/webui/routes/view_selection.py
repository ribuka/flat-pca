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
    file: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Replace the chosen files.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    file : list[str] | None, default None
        Checked stems. Stems that are not transform targets of the run in
        use are ignored, and only the first ``ui.explore_max_files`` of the
        rest (in option order) are kept.

    Returns
    -------
    Response
        Empty response reloading the page.
    """
    choice = current_view_choice(workspace)
    wanted = set(file or ())
    stems = [stem for stem in choice.file_options if stem in wanted]
    workspace.view_selection.replace_stems(stems[: workspace.settings.ui.explore_max_files])
    return _reload()


@router.post("/files/add")
def add_file(workspace: WorkspaceDependency, stem: Annotated[str, Form()]) -> JSONResponse:
    """Add one file to the chosen files, as a click on a score point does.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    stem : str
        Transform target of the run in use.

    Returns
    -------
    JSONResponse
        ``{"added": true}`` if the file is chosen afterwards; otherwise
        ``{"added": false, "message": ...}`` because the limit of chosen
        files is reached. The refusal is not an HTTP error, so the browser
        logs no failed request.

    Raises
    ------
    HTTPException
        With status 400 if ``stem`` is not a transform target of the run.
    """
    choice = current_view_choice(workspace)
    if stem not in choice.file_options:
        raise HTTPException(status_code=400, detail=f"not a transform target: {stem}")
    max_files = workspace.settings.ui.explore_max_files
    # Count only the chosen files that the run in use still has.
    workspace.view_selection.replace_stems(choice.files)
    if workspace.view_selection.add_stem(stem, max_files):
        return JSONResponse({"added": True})
    return JSONResponse(
        {
            "added": False,
            "message": f"表示ファイルは {max_files} 件まで選べます。"
            f"{stem} を追加するには、サイドバーでほかのファイルの選択を外してください。",
        }
    )
