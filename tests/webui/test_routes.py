"""Tests for the Web UI routes through FastAPI's ``TestClient``."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from flat_pca.webui.app import create_app
from flat_pca.webui.routes import catalog as catalog_routes
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


def _opening_tag(html: str, element_id: str) -> str:
    """Return the opening tag of the element with ``element_id``."""
    match = re.search(rf'<[a-z]+ id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return match.group()


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
    assert 'hx-get="/catalog/files"' in response.text
    assert "catalog はまだ作成されていません" in response.text


def test_data_selection_groups_are_collapsible(client: TestClient) -> None:
    """Each group of the page is an open accordion, without check-all buttons."""
    text = client.get("/").text

    for group in ("catalog", "files"):
        assert f'<details class="card" data-group="{group}" open>' in text
    assert 'data-group="filters"' not in text
    assert '<button type="button" data-check-all' not in text
    assert "data-uncheck-all" not in text


def test_static_scripts_are_served_locally(client: TestClient) -> None:
    """htmx and Plotly.js are served without a CDN."""
    assert client.get("/static/vendor/htmx.min.js").status_code == 200
    plotly = client.get("/static/vendor/plotly.min.js")
    assert plotly.status_code == 200
    assert plotly.headers["content-type"].startswith("text/javascript")


def test_icon_font_is_served_locally(client: TestClient) -> None:
    """The Material Symbols subset font is bundled and declared in app.css."""
    font = client.get("/static/vendor/material-symbols-outlined.woff2")
    assert font.status_code == 200
    assert font.content.startswith(b"wOF2")
    css = client.get("/static/app.css").text
    assert 'url("vendor/material-symbols-outlined.woff2")' in css
    assert "font-display: block" in css


def test_refresh_returns_polling_status_until_finished(
    client: TestClient, wait_for: Wait
) -> None:
    """Refreshing queues a run whose status partial polls every second."""
    response = client.post("/catalog/refresh")

    assert response.status_code == 200
    assert response.headers["HX-Trigger"] == "catalog-started"
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
    assert text.count("data-check-all") == 1
    assert text.index("data-check-all") < text.index("<tbody>")
    assert "全 3 件中 1〜3 件" in text
    assert "data-page=" not in text


def test_file_table_has_column_sort_and_filters(cataloged_client: TestClient) -> None:
    """Every column name sorts; the stem and metadata columns have filters."""
    text = cataloged_client.get("/catalog/files", params={"sort": "yield_pct"}).text

    for name in ("stem", "lot", "yield_pct", "date", "n_steps", "n_segments", "n_rows"):
        assert f'data-sort="{name}"' in text
    assert text.count('aria-sort="ascending"') == 1
    assert '<input type="hidden" name="sort" value="yield_pct" data-file-query>' in text
    assert '<input type="hidden" name="order" value="asc" data-file-query>' in text
    assert '<input type="hidden" name="page" value="1" data-file-query>' in text
    assert 'name="q"' in text
    assert 'name="eq__lot"' in text
    assert 'name="min__yield_pct"' in text
    assert 'name="max__yield_pct"' in text
    assert re.search(r'type="datetime-local" id="[^"]+" name="max__date"', text)
    assert "min__n_rows" not in text
    descending = cataloged_client.get(
        "/catalog/files", params={"sort": "n_rows", "order": "desc"}
    ).text
    assert descending.count('aria-sort="descending"') == 1


def test_file_table_category_choices_follow_catalog(
    client: TestClient, wait_for: Wait
) -> None:
    """Category choices come from the catalog and keep the chosen value."""
    assert '<option value="A"' not in client.get("/catalog/files").text
    workspace = _workspace(client)
    wait_for(workspace.database, workspace.submit_catalog())

    response = client.get("/catalog/files", params={"eq__lot": "B"})

    assert response.status_code == 200
    assert '<option value="A" >' in response.text
    assert '<option value="B" selected>' in response.text
    stale = client.get("/catalog/files", params={"eq__lot": "Z"}).text
    assert '<option value="Z" selected>' in stale


def test_file_table_pages_and_lists_matching_stems(
    cataloged_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files split into pages; every file matching the filters is listed."""
    monkeypatch.setattr(catalog_routes, "FILE_PAGE_SIZE", 2)

    first = cataloged_client.get("/catalog/files").text
    second = cataloged_client.get("/catalog/files", params={"page": "2"}).text
    past = cataloged_client.get("/catalog/files", params={"page": "9"}).text

    assert 'data-stem="run-2"' in first
    assert 'data-stem="run-10"' not in first
    assert "全 3 件中 1〜2 件" in first
    assert '<button type="button" data-page="2" >次へ</button>' in first
    assert 'data-stem="run-10"' in second
    assert 'data-stem="run-1"' not in second
    assert "全 3 件中 3〜3 件" in second
    assert 'aria-current="page" disabled>2</button>' in second
    assert "全 3 件中 3〜3 件" in past
    for text in (first, second):
        assert '["run-1", "run-2", "run-10"]</script>' in text
    filtered = cataloged_client.get("/catalog/files", params={"eq__lot": "B"}).text
    assert '["run-2"]</script>' in filtered
    assert cataloged_client.get("/catalog/files", params={"page": "0"}).status_code == 400


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
    assert response.headers["HX-Trigger"] == "selection-updated"
    assert 'data-selected-count="2"' in response.text
    assert ">check_circle</span>" in response.text
    assert ">check_circle</span>" not in cataloged_client.get("/").text
    assert _workspace(cataloged_client).selection.stems == ["run-2", "run-10"]
    table = cataloged_client.get("/catalog/files").text
    assert re.search(r'value="run-2"[^>]*checked', table)
    assert not re.search(r'value="run-1"[^>]*checked', table)
    page = cataloged_client.get("/").text
    assert re.search(
        r'id="selected-stems"[^>]*>'
        r'<input type="hidden" name="stems" value="run-2">'
        r'<input type="hidden" name="stems" value="run-10"></div>',
        page,
    )

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


def test_page_has_sidebar_navigation(client: TestClient) -> None:
    """The sidebar marks the current screen and links every screen."""
    page = client.get("/").text

    assert '<a class="nav-item nav-current" href="/" aria-current="page">' in page
    assert '<a class="nav-item" href="/fit">' in page
    assert '<a class="nav-item" href="/model">' in page
    assert '<a class="nav-item" href="/explore">' in page
    assert '<a class="nav-item" href="/scores">' in page
    assert '<a class="nav-item" href="/monitoring">' in page
    assert 'aria-disabled="true"' not in page
    sidebar = _opening_tag(page, "sidebar-status")
    assert 'hx-get="/sidebar/status"' in sidebar
    assert 'hx-target="this"' in sidebar
    for event in ("catalog-started", "catalog-updated", "selection-updated"):
        assert f"{event} from:body" in sidebar


def test_file_table_and_select_set_their_own_targets(client: TestClient) -> None:
    """The file table reloads itself; Select sends the kept selection."""
    page = client.get("/").text

    file_table = _opening_tag(page, "file-table")
    assert 'hx-target="this"' in file_table
    assert 'hx-swap="innerHTML"' in file_table
    select = _opening_tag(page, "select-files")
    assert 'hx-include="#selected-stems"' in select
    assert 'hx-target="#selection-summary"' in select
    assert 'hx-swap="outerHTML"' in select


def test_sidebar_status_shows_catalog_and_selection(
    client: TestClient, wait_for: Wait
) -> None:
    """The sidebar reports the catalog state and the selection size."""
    empty = client.get("/sidebar/status")
    assert empty.status_code == 200
    assert "未作成" in empty.text
    assert "every 2s" not in empty.text

    workspace = _workspace(client)
    run_id = workspace.submit_catalog()
    active = client.get("/sidebar/status").text
    assert 'hx-trigger="every 2s"' in active
    wait_for(workspace.database, run_id)
    client.post("/catalog/selection", data={"stems": ["run-1", "run-2"]})

    finished = client.get("/sidebar/status").text

    assert "status-succeeded" in finished
    assert "3 ファイル" in finished
    assert "every 2s" not in finished
    assert re.search(r'data-sidebar="selection">\s*2 ファイル', finished)
