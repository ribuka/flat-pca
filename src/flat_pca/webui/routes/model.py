"""Model screen: figures fixed by the fit run alone."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from flat_pca.visualize import create_loading_scatter, create_scree_plot

from ..services.display_cache import ShownMatrices
from ..services.model import (
    AGGREGATION_LABELS,
    HeatmapRequest,
    ModelRequest,
    ModelView,
    heatmap_request,
    resolve_model,
)
from ..services.runs import list_succeeded_runs
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace
from .heatmap_trend import heatmap_context, script_json, trend_response
from .view_selection import current_view_choice

router = APIRouter(prefix="/model")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _resolve(workspace: Workspace, request: ModelRequest) -> ModelView:
    """Resolve the requested screen, rejecting invalid parameters.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    request : ModelRequest
        Requested choices.

    Returns
    -------
    ModelView
        Resolved screen.

    Raises
    ------
    HTTPException
        With status 400 if a parameter is invalid.
    """
    try:
        return resolve_model(
            workspace.cache,
            list_succeeded_runs(workspace.database, FIT_JOB, request.run),
            request,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _figures(shown: ModelView) -> dict[str, str]:
    """Build the scree and loading figures of a resolved screen.

    Parameters
    ----------
    shown : ModelView
        Resolved screen without an error.

    Returns
    -------
    dict[str, str]
        Figure JSON keyed by ``scree`` and ``loadings``.
    """
    assert shown.explained_variance is not None
    assert shown.loadings is not None
    figures = {
        "scree": create_scree_plot(shown.explained_variance).update_layout(
            title="Scree plot"
        ),
        "loadings": create_loading_scatter(
            shown.loadings, x=shown.x_name, y=shown.y_name
        ).update_layout(title=f"Loadings ({AGGREGATION_LABELS[shown.aggregation]})"),
    }
    return {name: script_json(figure.to_json()) for name, figure in figures.items()}


def _trend_url(key: HeatmapRequest) -> str:
    """Return the trend URL of the heatmap.

    Parameters
    ----------
    key : HeatmapRequest
        Run, values, component, and segment of the drawn matrix.

    Returns
    -------
    str
        ``/model/trend`` with the query selecting the same matrix.
    """
    return "/model/trend?" + urlencode(
        {"run": key.run, "view": key.view, "k": key.component, "segment": key.segment}
    )


@router.get("", response_class=HTMLResponse)
def model_page(
    request: Request,
    workspace: WorkspaceDependency,
    x: int | None = None,
    y: int | None = None,
    aggregation: str | None = None,
    view: str | None = None,
    k: int | None = None,
    segment: str | None = None,
) -> HTMLResponse:
    """Render the model page of the fit run chosen in the sidebar.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    x : int | None, default None
        1-based component number m of the loading plot's horizontal axis;
        1 by default.
    y : int | None, default None
        1-based component number n of the loading plot's vertical axis;
        2 by default.
    aggregation : str | None, default None
        ``"mean"``, ``"rms"``, or ``"abs_mean"`` loading aggregation;
        ``"rms"`` by default.
    view : str | None, default None
        Values of the heatmap: ``"component"`` (component k) or a
        preprocessing parameter key; ``"component"`` by default and when
        the run lacks the parameter.
    k : int | None, default None
        1-based component number k of the component heatmap; 1 by default.
    segment : str | None, default None
        ``"{Step}:{Sequence}"`` of the heatmap; the first one by default.

    Returns
    -------
    HTMLResponse
        Full page with the figures.
    """
    shown = _resolve(
        workspace,
        ModelRequest(
            run=current_view_choice(workspace).run_id,
            x=x,
            y=y,
            aggregation=aggregation,
            view=view,
            component=k,
            segment=segment,
        ),
    )
    context: dict[str, object] = {
        "shown": shown,
        "aggregation_labels": AGGREGATION_LABELS,
    }
    key = heatmap_request(shown)
    if shown.error is None and key is not None:
        context["figures"] = _figures(shown)
        workspace.cache.keep_shown_matrices(
            key, ShownMatrices(value_name=shown.value_name, matrices=shown.matrices)
        )
        context |= heatmap_context(
            shown.matrices,
            shown.value_name,
            workspace.settings.ui.heatmap_max_cells,
            _trend_url(key),
        )
    return templates.TemplateResponse(request, "pages/model.html", context)


@router.get("/trend")
def model_trend(
    workspace: WorkspaceDependency,
    wavelength: float,
    step_time: float,
    run: str | None = None,
    view: str | None = None,
    k: int | None = None,
    segment: str | None = None,
) -> Response:
    """Return the trend figures of the heatmap at one point.

    The matrix kept by the page is used when possible; otherwise the screen
    is resolved and its matrix is kept under its resolved choices.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    wavelength : float
        Requested wavelength; the nearest one is used.
    step_time : float
        Requested ``StepTime``; the nearest one is used.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    view, k, segment
        Same as ``model_page``.

    Returns
    -------
    Response
        JSON of ``trend_response``.

    Raises
    ------
    HTTPException
        With status 400 if a parameter is invalid, or 404 if the screen has
        nothing to show.
    """
    kept = None
    if run is not None and view is not None and k is not None and segment is not None:
        kept = workspace.cache.shown_matrices(
            HeatmapRequest(run=run, view=view, component=k, segment=segment)
        )
    if kept is None:
        shown = _resolve(
            workspace, ModelRequest(run=run, view=view, component=k, segment=segment)
        )
        key = heatmap_request(shown)
        if key is None:
            raise HTTPException(status_code=404, detail=shown.error or "nothing to show")
        kept = ShownMatrices(value_name=shown.value_name, matrices=shown.matrices)
        workspace.cache.keep_shown_matrices(key, kept)
    return trend_response(kept.matrices, kept.value_name, wavelength, step_time)
