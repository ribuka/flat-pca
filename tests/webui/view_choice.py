"""Choosing the shown transform run and the shown files in route tests."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from flat_pca.webui.routes.view_selection import current_view_choice


def choose_view(
    client: TestClient, run: str | None = None, files: Sequence[str] | None = None
) -> None:
    """Choose the shown transform run and the shown files as the browser does.

    Parameters
    ----------
    client : TestClient
        Client of the application.
    run : str | None, default None
        Transform run to show, chosen with "Show" on the transform screen;
        ``None`` keeps the current choice. Choosing another run clears the
        shown files.
    files : Sequence[str] | None, default None
        Shown files to choose after the run; ``None`` keeps them. They are
        sent with the run in use, as the sidebar drawn with it does.
    """
    if run is not None:
        response = client.post("/transform/show", data={"run": run})
        assert response.status_code == 200
    if files is not None:
        shown_run = current_view_choice(client.app.state.workspace).run_id
        response = client.post(
            "/sidebar/selection/files", data={"run": shown_run, "file": list(files)}
        )
        assert response.status_code == 200
