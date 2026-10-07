"""Score screen: score scatter plot and partial score trajectories."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

import plotly.graph_objects as go
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from flat_pca.visualize import create_partial_score_trajectories, create_score_scatter

from ..services.runs import list_succeeded_runs
from ..services.scores import ScoresRequest, ScoresView, resolve_scores
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/scores")
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


def _figures(shown: ScoresView) -> dict[str, str]:
    """Build the figures of a resolved score screen.

    Parameters
    ----------
    shown : ScoresView
        Resolved screen without an error.

    Returns
    -------
    dict[str, str]
        Figure JSON keyed by ``scatter`` and, when a chosen file has a
        trajectory, ``trajectories``.
    """
    assert shown.scores is not None
    figures = {
        "scatter": create_score_scatter(
            shown.scores.x,
            shown.scores.y,
            labels=shown.scores.samples["stem"].to_list(),
            x_name=shown.x_name,
            y_name=shown.y_name,
            color=None if shown.color is None else shown.scores.samples[shown.color],
        ).update_layout(title="スコア"),
    }
    if shown.trajectories:
        figures["trajectories"] = create_partial_score_trajectories(
            {
                stem: (trajectory.x, trajectory.y, trajectory.points)
                for stem, trajectory in shown.trajectories.items()
            },
            x_name=shown.x_name,
            y_name=shown.y_name,
        ).update_layout(title="部分スコア軌跡")
    return {name: _figure_json(figure) for name, figure in figures.items()}


@router.get("", response_class=HTMLResponse)
def scores_page(
    request: Request,
    workspace: WorkspaceDependency,
    run: str | None = None,
    x: int | None = None,
    y: int | None = None,
    color: str | None = None,
    file: Annotated[list[str] | None, Query()] = None,
) -> HTMLResponse:
    """Render the score page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    run : str | None, default None
        Succeeded fit run; the latest one by default.
    x : int | None, default None
        1-based component number m of the horizontal axes; 1 by default.
    y : int | None, default None
        1-based component number n of the vertical axes; 2 by default.
    color : str | None, default None
        Metadata column coloring the score points; ``ui.default_color_by``
        by default and none for ``""``.
    file : list[str] | None, default None
        Stems whose partial score trajectories are drawn; the first file by
        default.

    Returns
    -------
    HTMLResponse
        Full page with the figures.

    Raises
    ------
    HTTPException
        With status 400 if the run is invalid.
    """
    scores_request = ScoresRequest(run=run, x=x, y=y, color=color, files=tuple(file or ()))
    try:
        shown = resolve_scores(
            workspace.cache,
            list_succeeded_runs(workspace.database, FIT_JOB, run),
            scores_request,
            workspace.settings.ui.default_color_by,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    context: dict[str, object] = {"shown": shown}
    if shown.error is None and shown.run_id is not None:
        context["figures"] = _figures(shown)
        context["explore_url"] = "/explore?" + urlencode(
            {"view": "preprocessed", "run": shown.run_id}
        )
    return templates.TemplateResponse(request, "pages/scores.html", context)
