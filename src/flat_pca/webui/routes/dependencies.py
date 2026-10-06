"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from fastapi import Request

from ..workspace import Workspace


def get_workspace(request: Request) -> Workspace:
    """Return the application's workspace.

    Parameters
    ----------
    request : Request
        Current request.

    Returns
    -------
    Workspace
        Workspace created in the application lifespan.
    """
    return request.app.state.workspace
