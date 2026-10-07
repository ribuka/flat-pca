"""Spectral exploration screen: heatmap and trends of one ``(Step, Sequence)``."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from ..services.display_cache import ShownMatrices
from ..services.explore import (
    VIEW_LABELS,
    ExploreRequest,
    ExploreView,
    resolve_explore,
    shown_request,
)
from ..services.runs import list_succeeded_runs
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace
from .heatmap_trend import heatmap_context, trend_response
from .view_selection import current_view_choice

router = APIRouter(prefix="/explore")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _resolve(workspace: Workspace, request: ExploreRequest) -> ExploreView:
    """Resolve the requested view, rejecting invalid parameters.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    request : ExploreRequest
        Requested choices.

    Returns
    -------
    ExploreView
        Resolved view.

    Raises
    ------
    HTTPException
        With status 400 if a parameter is invalid.
    """
    try:
        return resolve_explore(
            workspace.cache,
            list_succeeded_runs(workspace.database, FIT_JOB, request.run),
            request,
            workspace.settings.ui.explore_max_files,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _query_string(shown: ExploreView) -> str:
    """Encode the resolved choices of a view as query parameters.

    Parameters
    ----------
    shown : ExploreView
        Resolved view.

    Returns
    -------
    str
        Query string selecting the same view, so trend requests show the
        data of the drawn heatmap.
    """
    request = shown_request(shown)
    parameters = [("view", request.view)]
    if request.run is not None:
        parameters.append(("run", request.run))
    parameters.extend(("file", stem) for stem in request.files)
    if request.segment is not None:
        parameters.append(("segment", request.segment))
    if request.component is not None:
        parameters.append(("k", str(request.component)))
    return urlencode(parameters)


def _shown_matrices(workspace: Workspace, request: ExploreRequest) -> ShownMatrices:
    """Return the unbinned matrices of a view, kept from its page when possible.

    The page keeps the matrices of the view it draws, so the trend requests
    of that page are cut from them without resolving the view again. After
    they are evicted, or for a request no page drew, the view is resolved
    and its matrices are kept under its resolved choices, so a request
    relying on a default (such as the latest run) is always resolved.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    request : ExploreRequest
        Requested choices.

    Returns
    -------
    ShownMatrices
        Matrices and value name of the view.

    Raises
    ------
    HTTPException
        With status 400 if a parameter is invalid, or 404 if the view has
        nothing to show.
    """
    kept = workspace.cache.shown_matrices(request)
    if kept is not None:
        return kept
    shown = _resolve(workspace, request)
    if not shown.matrices:
        raise HTTPException(status_code=404, detail=shown.error or "nothing to show")
    kept = ShownMatrices(value_name=shown.value_name, matrices=shown.matrices)
    workspace.cache.keep_shown_matrices(shown_request(shown), kept)
    return kept


@router.get("", response_class=HTMLResponse)
def explore_page(
    request: Request,
    workspace: WorkspaceDependency,
    view: str = "raw",
    segment: str | None = None,
    k: int | None = None,
) -> HTMLResponse:
    """Render the spectral exploration page.

    The fit run and the files are those chosen in the sidebar; without
    chosen files, the first transform target is shown.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    view : str, default ``"raw"``
        A key of ``VIEW_LABELS``.
    segment : str | None, default None
        ``"{Step}:{Sequence}"``; the first available one by default.
    k : int | None, default None
        1-based component number k of the contribution, reconstruction,
        and residual views.

    Returns
    -------
    HTMLResponse
        Full page with the heatmap figure and its unbinned axes.
    """
    choice = current_view_choice(workspace)
    explore_request = ExploreRequest(
        view=view,
        run=choice.run_id,
        files=tuple(choice.files),
        segment=segment,
        component=k,
    )
    shown = _resolve(workspace, explore_request)
    context: dict[str, object] = {
        "shown": shown,
        "view_labels": VIEW_LABELS,
        "max_files": workspace.settings.ui.explore_max_files,
    }
    if shown.matrices:
        workspace.cache.keep_shown_matrices(
            shown_request(shown),
            ShownMatrices(value_name=shown.value_name, matrices=shown.matrices),
        )
        context |= heatmap_context(
            shown.matrices,
            shown.value_name,
            workspace.settings.ui.heatmap_max_cells,
            f"/explore/trend?{_query_string(shown)}",
        )
    return templates.TemplateResponse(request, "pages/explore.html", context)


@router.get("/trend")
def explore_trend(
    workspace: WorkspaceDependency,
    wavelength: float,
    step_time: float,
    view: str = "raw",
    run: str | None = None,
    file: Annotated[list[str] | None, Query()] = None,
    segment: str | None = None,
    k: int | None = None,
) -> Response:
    """Return the trend figures at one point, cut from the unbinned values.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    wavelength : float
        Requested wavelength; each trace uses its nearest one.
    step_time : float
        Requested ``StepTime``; each trace uses its nearest one.
    view, segment, k
        Same as ``explore_page``.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    file : list[str] | None, default None
        Stems to overlay; the first transform target by default.

    Returns
    -------
    Response
        JSON with ``wavelength`` and ``step_time`` (the heatmap trace's grid
        point), ``by_step_time`` (figure of the values over ``StepTime`` at
        the wavelength), and ``by_wavelength`` (figure of the values over
        wavelength at the ``StepTime``).

    Raises
    ------
    HTTPException
        With status 400 if a parameter is invalid, or 404 if the view has
        nothing to show.
    """
    shown = _shown_matrices(
        workspace,
        ExploreRequest(
            view=view, run=run, files=tuple(file or ()), segment=segment, component=k
        ),
    )
    return trend_response(shown.matrices, shown.value_name, wavelength, step_time)
