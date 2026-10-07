"""Choosing the sidebar's fit run and shown files in route tests."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient


def choose_view(
    client: TestClient, run: str | None = None, files: Sequence[str] | None = None
) -> None:
    """Choose the sidebar's fit run and shown files as the browser does.

    Parameters
    ----------
    client : TestClient
        Client of the application.
    run : str | None, default None
        Fit run to choose; ``None`` keeps the current choice. Choosing
        another run clears the shown files.
    files : Sequence[str] | None, default None
        Shown files to choose after the run; ``None`` keeps them.
    """
    if run is not None:
        response = client.post("/sidebar/selection/run", data={"run": run})
        assert response.status_code == 200
    if files is not None:
        response = client.post("/sidebar/selection/files", data={"file": list(files)})
        assert response.status_code == 200
