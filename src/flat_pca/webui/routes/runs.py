"""Run control shared by every job kind."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from ..services.runs import get_run
from ..workspace import Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/runs")


@router.post("/{run_id}/cancel", status_code=204)
def cancel_run(
    run_id: str, workspace: Annotated[Workspace, Depends(get_workspace)]
) -> Response:
    """Cancel a queued or running run.

    The run's state change is picked up by the status polling of the screen
    that shows it.

    Parameters
    ----------
    run_id : str
        Run to cancel.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    Response
        Empty response with status 204.

    Raises
    ------
    HTTPException
        With status 404 if the run does not exist, or 409 if it is neither
        queued nor running.
    """
    if get_run(workspace.database, run_id) is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    if not workspace.executor.cancel(run_id):
        raise HTTPException(status_code=409, detail=f"run is not active: {run_id}")
    return Response(status_code=204)
