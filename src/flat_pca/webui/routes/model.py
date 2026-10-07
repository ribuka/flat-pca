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
    COMPONENT_VALUE_NAME,
    ComponentRequest,
    ModelRequest,
    ModelView,
    component_request,
    resolve_model,
)
from ..services.runs import list_succeeded_runs
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace
from .heatmap_trend import heatmap_context, script_json, trend_response

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
            title="スクリープロット"
        ),
        "loadings": create_loading_scatter(
            shown.loadings, x=shown.x_name, y=shown.y_name
        ).update_layout(title=f"ローディング（{AGGREGATION_LABELS[shown.aggregation]}）"),
    }
    return {name: script_json(figure.to_json()) for name, figure in figures.items()}


def _trend_url(key: ComponentRequest) -> str:
    """Return the trend URL of the component heatmap.

    Parameters
    ----------
    key : ComponentRequest
        Run, component, and segment of the drawn matrix.

    Returns
    -------
    str
        ``/model/trend`` with the query selecting the same matrix.
    """
    return "/model/trend?" + urlencode(
        {"run": key.run, "k": key.component, "segment": key.segment}
    )


@router.get("", response_class=HTMLResponse)
def model_page(
    request: Request,
    workspace: WorkspaceDependency,
    run: str | None = None,
    x: int | None = None,
    y: int | None = None,
    aggregation: str | None = None,
    k: int | None = None,
    segment: str | None = None,
) -> HTMLResponse:
    """Render the model page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    x : int | None, default None
        1-based component number m of the loading plot's horizontal axis;
        1 by default.
    y : int | None, default None
        1-based component number n of the loading plot's vertical axis;
        2 by default.
    aggregation : str | None, default None
        ``"mean"``, ``"rms"``, or ``"abs_mean"`` loading aggregation;
        ``"rms"`` by default.
    k : int | None, default None
        1-based component number k of the component heatmap; 1 by default.
    segment : str | None, default None
        ``"{Step}:{Sequence}"`` of the component heatmap; the first one by
        default.

    Returns
    -------
    HTMLResponse
        Full page with the figures.
    """
    shown = _resolve(
        workspace,
        ModelRequest(
            run=run, x=x, y=y, aggregation=aggregation, component=k, segment=segment
        ),
    )
    context: dict[str, object] = {
        "shown": shown,
        "aggregation_labels": AGGREGATION_LABELS,
    }
    key = component_request(shown)
    if shown.error is None and key is not None:
        context["figures"] = _figures(shown)
        workspace.cache.keep_shown_matrices(
            key, ShownMatrices(value_name=COMPONENT_VALUE_NAME, matrices=shown.matrices)
        )
        context |= heatmap_context(
            shown.matrices,
            COMPONENT_VALUE_NAME,
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
    k: int | None = None,
    segment: str | None = None,
) -> Response:
    """Return the trend figures of the component heatmap at one point.

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
    run, k, segment
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
    if run is not None and k is not None and segment is not None:
        kept = workspace.cache.shown_matrices(
            ComponentRequest(run=run, component=k, segment=segment)
        )
    if kept is None:
        shown = _resolve(workspace, ModelRequest(run=run, component=k, segment=segment))
        key = component_request(shown)
        if key is None:
            raise HTTPException(status_code=404, detail=shown.error or "nothing to show")
        kept = ShownMatrices(value_name=COMPONENT_VALUE_NAME, matrices=shown.matrices)
        workspace.cache.keep_shown_matrices(key, kept)
    return trend_response(kept.matrices, kept.value_name, wavelength, step_time)
