"""T² and Q screen: control charts and the T² × Q scatter plot."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

import plotly.graph_objects as go
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from flat_pca.visualize import create_control_chart, create_t2_q_scatter

from ..services.monitoring import MonitoringRequest, MonitoringView, resolve_monitoring
from ..services.runs import list_succeeded_runs
from ..templating import format_value, templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/monitoring")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _figure_json(figure: go.Figure) -> str:
    """Serialize a figure for a ``<script type="application/json">`` element.

    Parameters
    ----------
    figure : go.Figure
        Figure to embed.

    Returns
    -------
    str
        Figure JSON with ``</`` escaped so it cannot close the element.
    """
    return figure.to_json().replace("</", "<\\/")


def _figures(shown: MonitoringView) -> dict[str, str]:
    """Build the figures of a resolved T² and Q screen.

    Parameters
    ----------
    shown : MonitoringView
        Resolved screen without an error.

    Returns
    -------
    dict[str, str]
        Figure JSON keyed by ``t2``, ``q``, and ``scatter``.
    """
    points = shown.points
    assert points is not None
    labels = points.samples["stem"].to_list()
    order_values = (
        None
        if shown.order is None
        else [format_value(value) for value in points.samples[shown.order].to_list()]
    )
    figures = {
        "t2": create_control_chart(
            points.t2,
            ucl=points.t2_ucl,
            labels=labels,
            y_name="T²",
            order_values=order_values,
            order_name=shown.order,
        ).update_layout(title="T² 管理図"),
        "q": create_control_chart(
            points.q,
            ucl=points.q_ucl,
            labels=labels,
            y_name="Q",
            order_values=order_values,
            order_name=shown.order,
        ).update_layout(title="Q 管理図"),
        "scatter": create_t2_q_scatter(
            points.t2,
            points.q,
            t2_ucl=points.t2_ucl,
            q_ucl=points.q_ucl,
            labels=labels,
        ).update_layout(title="T² × Q"),
    }
    return {name: _figure_json(figure) for name, figure in figures.items()}


@router.get("", response_class=HTMLResponse)
def monitoring_page(
    request: Request,
    workspace: WorkspaceDependency,
    run: str | None = None,
    order: str | None = None,
) -> HTMLResponse:
    """Render the T² and Q page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    order : str | None, default None
        Metadata column ordering the control charts; ``ui.default_order_by``
        by default and the natural order of the stems for ``""``.

    Returns
    -------
    HTMLResponse
        Full page with the figures.

    Raises
    ------
    HTTPException
        With status 400 if the run is invalid.
    """
    try:
        shown = resolve_monitoring(
            workspace.cache,
            list_succeeded_runs(workspace.database, FIT_JOB, run),
            MonitoringRequest(run=run, order=order),
            workspace.settings.ui.default_order_by,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    context: dict[str, object] = {"shown": shown}
    if shown.error is None and shown.run_id is not None:
        context["figures"] = _figures(shown)
        context["explore_url"] = "/explore?" + urlencode(
            {"view": "q_contribution", "run": shown.run_id}
        )
    return templates.TemplateResponse(request, "pages/monitoring.html", context)
