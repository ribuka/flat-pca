"""Spectral exploration screen: heatmap and trends of one ``(Step, Sequence)``."""

from __future__ import annotations

import json
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from flat_pca.visualize import create_heatmap, create_trend

from ..services.display_cache import ShownMatrices
from ..services.explore import (
    VIEW_LABELS,
    ExploreRequest,
    ExploreView,
    explore_trends,
    resolve_explore,
    shown_request,
)
from ..services.heatmap_binning import bin_step_times
from ..services.runs import list_succeeded_runs
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace

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
            workspace.database,
            workspace.selection.stems,
            workspace.cache,
            list_succeeded_runs(workspace.database, FIT_JOB, request.run),
            request,
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


def _script_json(text: str) -> str:
    """Make JSON text safe to embed in a ``<script>`` element.

    Parameters
    ----------
    text : str
        JSON text.

    Returns
    -------
    str
        ``text`` with ``</`` escaped so it cannot close the element.
    """
    return text.replace("</", "<\\/")


@router.get("", response_class=HTMLResponse)
def explore_page(
    request: Request,
    workspace: WorkspaceDependency,
    view: str = "raw",
    run: str | None = None,
    file: Annotated[list[str] | None, Query()] = None,
    segment: str | None = None,
    k: int | None = None,
) -> HTMLResponse:
    """Render the spectral exploration page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    view : str, default ``"raw"``
        A key of ``VIEW_LABELS``.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    file : list[str] | None, default None
        Stems to show; the first available one by default.
    segment : str | None, default None
        ``"{Step}:{Sequence}"``; the first available one by default.
    k : int | None, default None
        1-based component number k of the component and reconstruction
        views.

    Returns
    -------
    HTMLResponse
        Full page with the heatmap figure and its unbinned axes.
    """
    explore_request = ExploreRequest(
        view=view, run=run, files=tuple(file or ()), segment=segment, component=k
    )
    shown = _resolve(workspace, explore_request)
    context: dict[str, object] = {
        "shown": shown,
        "view_labels": VIEW_LABELS,
        "max_cells": workspace.settings.ui.heatmap_max_cells,
    }
    if shown.matrices:
        workspace.cache.keep_shown_matrices(
            shown_request(shown),
            ShownMatrices(value_name=shown.value_name, matrices=shown.matrices),
        )
        label, matrix = next(iter(shown.matrices.items()))
        binned = bin_step_times(matrix, workspace.settings.ui.heatmap_max_cells)
        figure = create_heatmap(
            z=binned.matrix.values,
            x=binned.matrix.wavelengths,
            y=binned.matrix.step_times,
            y_name="StepTime",
            z_name=shown.value_name,
        ).update_layout(title=label)
        context["heatmap_label"] = label
        context["binned"] = binned
        context["figure_json"] = _script_json(figure.to_json())
        context["axes_json"] = _script_json(
            json.dumps(
                {
                    "wavelengths": matrix.wavelengths.tolist(),
                    "step_times": matrix.step_times.tolist(),
                }
            )
        )
        context["n_wavelengths"] = matrix.wavelengths.size
        context["n_step_times"] = matrix.step_times.size
        context["trend_query"] = _query_string(shown)
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
    view, run, file, segment, k
        Same as ``explore_page``.

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
    by_time, by_wavelength = explore_trends(shown.matrices, wavelength, step_time)
    first_time = next(iter(by_time.values()))
    first_wavelength = next(iter(by_wavelength.values()))
    time_figure = create_trend(
        {label: (line.x, line.y) for label, line in by_time.items()},
        x_name="StepTime",
        y_name=shown.value_name,
        title=f"wavelength = {first_time.at:g}",
    )
    wavelength_figure = create_trend(
        {label: (line.x, line.y) for label, line in by_wavelength.items()},
        x_name="wavelength",
        y_name=shown.value_name,
        title=f"StepTime = {first_wavelength.at:g}",
    )
    body = (
        f'{{"wavelength": {json.dumps(first_time.at)}, '
        f'"step_time": {json.dumps(first_wavelength.at)}, '
        f'"by_step_time": {time_figure.to_json()}, '
        f'"by_wavelength": {wavelength_figure.to_json()}}}'
    )
    return Response(content=body, media_type="application/json")
