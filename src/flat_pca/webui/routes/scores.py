"""Score screen: score scatter plot and partial score trajectories."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from flat_pca.visualize import create_partial_score_trajectories, create_score_scatter

from ..services.point_table import SELECT_TOOLS, PointTable, point_table
from ..services.scores import ScoresRequest, ScoresView, resolve_scores
from ..templating import script_json
from ..workspace import Workspace
from .dependencies import get_workspace
from .view_page import render_view_page
from .view_selection import current_view_choice

router = APIRouter(prefix="/scores")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


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
        ).update_layout(title="Scores", modebar_add=SELECT_TOOLS),
    }
    if shown.trajectories:
        figures["trajectories"] = create_partial_score_trajectories(
            {
                stem: (trajectory.x, trajectory.y, trajectory.points)
                for stem, trajectory in shown.trajectories.items()
            },
            x_name=shown.trajectory_x_name,
            y_name=shown.trajectory_y_name,
            color=shown.trajectory_colors,
        ).update_layout(title="Partial score trajectories")
    return {name: script_json(figure.to_json()) for name, figure in figures.items()}


def _point_table(shown: ScoresView) -> PointTable:
    """Build the table of the files drawn in a resolved score scatter plot.

    Parameters
    ----------
    shown : ScoresView
        Resolved screen without an error.

    Returns
    -------
    PointTable
        One row per scored file with the shown scores of PCm and PCn.
    """
    scores = shown.scores
    assert scores is not None
    return point_table(
        scores.samples, {shown.x_name: scores.x.tolist(), shown.y_name: scores.y.tolist()}
    )


@router.get("", response_class=HTMLResponse)
def scores_page(
    request: Request,
    workspace: WorkspaceDependency,
    x: int | None = None,
    y: int | None = None,
    color: str | None = None,
    trajectory_x: int | None = None,
    trajectory_y: int | None = None,
    trajectory_color: str | None = None,
) -> HTMLResponse:
    """Render the score page.

    The fit run and the files of the partial score trajectories are those
    chosen in the sidebar; without chosen files, the first transform target
    is drawn.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    x : int | None, default None
        1-based component number m of the score scatter plot's horizontal
        axis; 1 by default.
    y : int | None, default None
        1-based component number n of the score scatter plot's vertical
        axis; 2 by default.
    color : str | None, default None
        Column (``stem`` or a metadata column) coloring the score points;
        ``ui.default_color_by`` by default and none for ``""``.
    trajectory_x : int | None, default None
        1-based component number m of the trajectories' horizontal axis; 1
        by default.
    trajectory_y : int | None, default None
        1-based component number n of the trajectories' vertical axis; 2 by
        default.
    trajectory_color : str | None, default None
        Column coloring the trajectories; ``stem`` by default and none for
        ``""``.

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
    scores_request = ScoresRequest(
        run=choice.run_id,
        x=x,
        y=y,
        color=color,
        files=tuple(choice.files),
        trajectory_x=trajectory_x,
        trajectory_y=trajectory_y,
        trajectory_color=trajectory_color,
    )
    try:
        shown = resolve_scores(
            workspace.cache,
            choice.runs,
            scores_request,
            workspace.settings.ui.default_color_by,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    context: dict[str, object] = {"shown": shown}
    if shown.error is None and shown.run_id is not None:
        context["figures"] = _figures(shown)
        context["point_table"] = _point_table(shown)
    return render_view_page(request, "pages/scores.html", context)
