"""T² and Q screen: control charts and the T² × Q scatter plot."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from flat_pca.visualize import create_control_chart, create_t2_q_scatter

from ..datatable import format_value
from ..services.monitoring import MonitoringRequest, MonitoringView, resolve_monitoring
from ..services.point_table import SELECT_TOOLS, PointTable, point_table
from ..templating import script_json
from ..workspace import Workspace
from .dependencies import get_workspace
from .view_page import render_view_page
from .view_selection import current_view_choice

router = APIRouter(prefix="/monitoring")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _figures(shown: MonitoringView) -> dict[str, str]:
    """Build the figures of a resolved T² and Q screen.

    Every figure gets the values of all scored files, so the colors match;
    the control charts draw only the files of ``MonitoringView.chart_rows``.

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
    samples = points.samples
    x_axis = shown.x_axis
    labels = samples["stem"].to_list()
    color = None if shown.color is None else samples[shown.color]
    chart: dict[str, Any] = {
        "labels": labels,
        "x": samples[x_axis] if shown.numeric_axis and x_axis is not None else None,
        "x_text": None
        if x_axis is None
        else [format_value(value) for value in samples[x_axis].to_list()],
        "x_name": x_axis,
        "color": color,
        "rows": shown.chart_rows,
    }
    figures = {
        "t2": create_control_chart(
            points.t2, ucl=points.t2_ucl, y_name="T²", **chart
        ).update_layout(title="T² control chart"),
        "q": create_control_chart(
            points.q, ucl=points.q_ucl, y_name="Q", **chart
        ).update_layout(title="Q control chart"),
        "scatter": create_t2_q_scatter(
            points.t2,
            points.q,
            t2_ucl=points.t2_ucl,
            q_ucl=points.q_ucl,
            labels=labels,
            color=color,
        ).update_layout(title="T² × Q"),
    }
    return {
        name: script_json(figure.update_layout(modebar_add=SELECT_TOOLS).to_json())
        for name, figure in figures.items()
    }


def _point_table(shown: MonitoringView) -> PointTable:
    """Build the table of the files drawn on a resolved T² and Q screen.

    Parameters
    ----------
    shown : MonitoringView
        Resolved screen without an error.

    Returns
    -------
    PointTable
        One row per scored file, in control-chart order, with T², Q, and
        whether each exceeds its UCL.
    """
    points = shown.points
    assert points is not None
    return point_table(
        points.samples,
        {
            "T²": points.t2.tolist(),
            "Q": points.q.tolist(),
            "T² above UCL": ["yes" if value else "" for value in points.t2 > points.t2_ucl],
            "Q above UCL": ["yes" if value else "" for value in points.q > points.q_ucl],
        },
    )


@router.get("", response_class=HTMLResponse)
def monitoring_page(
    request: Request,
    workspace: WorkspaceDependency,
    x_axis: str | None = None,
    as_category: bool = False,
    color: str | None = None,
) -> HTMLResponse:
    """Render the T² and Q page of the fit run chosen in the sidebar.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    x_axis : str | None, default None
        Metadata column on the horizontal axis of the control charts;
        ``ui.default_x_axis`` by default and the natural order of the stems
        for ``""``.
    as_category : bool, default False
        Whether a numeric, date, or datetime ``x_axis`` is drawn by rank
        instead of on a numeric axis.
    color : str | None, default None
        Metadata column coloring the points of all three figures;
        ``ui.default_color_by`` by default and none for ``""``.

    Returns
    -------
    HTMLResponse
        Full page with the figures, or its main part for an htmx request
        (see ``render_view_page``).

    Raises
    ------
    HTTPException
        With status 400 if the run is invalid.
    """
    choice = current_view_choice(workspace)
    try:
        shown = resolve_monitoring(
            workspace.cache,
            choice.runs,
            MonitoringRequest(
                run=choice.run_id, x_axis=x_axis, as_category=as_category, color=color
            ),
            workspace.settings.ui.default_x_axis,
            workspace.settings.ui.default_color_by,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    context: dict[str, object] = {"shown": shown}
    if shown.error is None and shown.run_id is not None:
        context["figures"] = _figures(shown)
        context["point_table"] = _point_table(shown)
    return render_view_page(request, "pages/monitoring.html", context)
