"""Tests for the Web UI routes through FastAPI's ``TestClient``."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from flat_pca.webui.app import create_app
from flat_pca.webui.services.runs import latest_run
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import CATALOG_JOB, Workspace

Wait = Callable[..., dict[str, object]]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application, opening its workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


@pytest.fixture
def cataloged_client(client: TestClient, wait_for: Wait) -> TestClient:
    """Return a client after one successful catalog run."""
    workspace = _workspace(client)
    assert (
        wait_for(workspace.database, workspace.submit_catalog())["status"]
        == "succeeded"
    )
    return client


def test_data_selection_page_renders_controls(client: TestClient) -> None:
    """The page links the bundled scripts and shows the catalog controls."""
    response = client.get("/")

    assert response.status_code == 200
    assert "/static/vendor/htmx.min.js" in response.text
    assert 'hx-post="/catalog/refresh"' in response.text
    assert 'name="eq__lot"' in response.text
    assert 'name="min__yield_pct"' in response.text
    assert 'type="datetime-local" name="max__date"' in response.text
    assert "catalog はまだ作成されていません" in response.text


def test_static_scripts_are_served_locally(client: TestClient) -> None:
    """htmx and Plotly.js are served without a CDN."""
    assert client.get("/static/vendor/htmx.min.js").status_code == 200
    plotly = client.get("/static/vendor/plotly.min.js")
    assert plotly.status_code == 200
    assert plotly.headers["content-type"].startswith("text/javascript")


def test_refresh_returns_polling_status_until_finished(
    client: TestClient, wait_for: Wait
) -> None:
    """Refreshing queues a run whose status partial polls every second."""
    response = client.post("/catalog/refresh")

    assert response.status_code == 200
    assert 'id="catalog-status"' in response.text
    assert 'hx-trigger="every 1s"' in response.text
    workspace = _workspace(client)
    run = latest_run(workspace.database, CATALOG_JOB)
    assert run is not None
    wait_for(workspace.database, str(run["run_id"]))

    finished = client.get("/catalog/status", params={"polling": "true"})

    assert finished.status_code == 200
    assert finished.headers["HX-Trigger"] == "catalog-updated"
    assert 'data-run-status="succeeded"' in finished.text
    assert "every 1s" not in finished.text
    assert "HX-Trigger" not in client.get("/catalog/status").headers


def test_file_table_lists_files_and_metadata_warnings(
    cataloged_client: TestClient,
) -> None:
    """The table lists every file and the CSV mismatches."""
    response = cataloged_client.get("/catalog/files")

    assert response.status_code == 200
    text = response.text
    assert text.index('data-stem="run-1"') < text.index('data-stem="run-2"')
    assert text.index('data-stem="run-2"') < text.index('data-stem="run-10"')
    assert "2026-01-02 03:04:05" in text
    assert 'data-warning="files-without-metadata"' in text
    assert "<li>ghost</li>" in text


def test_category_filters_follow_catalog_and_keep_selection(
    client: TestClient, wait_for: Wait
) -> None:
    """Category choices reload after a catalog update, keeping chosen values."""
    page = client.get("/").text
    span = re.search(r'<span id="category-filters"[^>]*>', page)
    assert span is not None
    # The span sits inside #file-filter, whose hx-target would otherwise be
    # inherited and swap the choices into the file table.
    assert 'hx-trigger="catalog-updated from:body"' in span.group()
    assert 'hx-target="this"' in span.group()
    assert '<option value="A"' not in page
    workspace = _workspace(client)
    wait_for(workspace.database, workspace.submit_catalog())

    response = client.get("/catalog/category-filters", params={"eq__lot": "B"})

    assert response.status_code == 200
    assert '<option value="A" >' in response.text
    assert '<option value="B" selected>' in response.text
    stale = client.get("/catalog/category-filters", params={"eq__lot": "Z"}).text
    assert '<option value="Z" selected>' in stale
    assert (
        client.get("/catalog/category-filters", params={"eq__unknown": "x"}).status_code
        == 400
    )


def test_file_table_applies_filters(cataloged_client: TestClient) -> None:
    """Query parameters filter the table."""
    response = cataloged_client.get(
        "/catalog/files", params={"eq__lot": "A", "sort": "yield_pct", "order": "desc"}
    )

    assert response.status_code == 200
    assert 'data-stem="run-1"' in response.text
    assert 'data-stem="run-2"' not in response.text


def test_file_table_rejects_invalid_filter(cataloged_client: TestClient) -> None:
    """Invalid filters are a client error."""
    response = cataloged_client.get("/catalog/files", params={"min__lot": "1"})

    assert response.status_code == 400


def test_selection_keeps_cataloged_stems(cataloged_client: TestClient) -> None:
    """The selection ignores unknown stems and is reflected in the table."""
    response = cataloged_client.post(
        "/catalog/selection", data={"stems": ["run-2", "ghost", "run-10"]}
    )

    assert response.status_code == 200
    assert 'data-selected-count="2"' in response.text
    assert _workspace(cataloged_client).selection.stems == ["run-2", "run-10"]
    table = cataloged_client.get("/catalog/files").text
    assert 'value="run-2" checked' in table
    assert 'value="run-1" checked' not in table

    cleared = cataloged_client.post("/catalog/selection")

    assert 'data-selected-count="0"' in cleared.text


def test_cancel_rejects_unknown_and_finished_runs(cataloged_client: TestClient) -> None:
    """Only active runs can be cancelled."""
    run = latest_run(_workspace(cataloged_client).database, CATALOG_JOB)
    assert run is not None

    assert cataloged_client.post("/runs/missing/cancel").status_code == 404
    assert cataloged_client.post(f"/runs/{run['run_id']}/cancel").status_code == 409


def test_cancel_active_run(client: TestClient, wait_for: Wait) -> None:
    """Cancelling an active catalog run returns 204 and the run ends cancelled."""
    workspace = _workspace(client)
    run_id = workspace.submit_catalog()

    response = client.post(f"/runs/{run_id}/cancel")

    assert response.status_code == 204
    assert wait_for(workspace.database, run_id)["status"] == "cancelled"
