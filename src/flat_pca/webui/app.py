"""FastAPI application factory of the Web UI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import catalog, runs
from .settings import Settings
from .templating import STATIC_DIR
from .workspace import Workspace

PLOTLY_JS = Path(str(files("plotly") / "package_data" / "plotly.min.js"))


def plotly_js() -> FileResponse:
    """Serve the Plotly.js bundle shipped with the ``plotly`` package.

    Serving the package's own copy keeps the browser library in step with
    the Python ``plotly`` version without a CDN.

    Returns
    -------
    FileResponse
        ``plotly.min.js``.
    """
    return FileResponse(PLOTLY_JS, media_type="text/javascript")


def create_app(settings: Settings) -> FastAPI:
    """Create the Web UI application.

    The ``Workspace`` is opened when the application starts and closed when
    it stops, and is available as ``app.state.workspace``.

    Parameters
    ----------
    settings : Settings
        Validated application settings.

    Returns
    -------
    FastAPI
        Configured application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the workspace for the application's lifetime."""
        workspace = Workspace(settings)
        app.state.workspace = workspace
        try:
            yield
        finally:
            workspace.close()

    app = FastAPI(title="flat-pca", lifespan=lifespan)
    app.add_api_route(
        "/static/vendor/plotly.min.js", plotly_js, include_in_schema=False
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(catalog.router)
    app.include_router(runs.router)
    return app
