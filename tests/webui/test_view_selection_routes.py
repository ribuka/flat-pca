"""Tests for the shown transform run and the sidebar's file choices."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fit_runs import register_shown_run
from spectra import SPECTRA_FILE_COUNT
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.routes import view_selection as view_selection_routes
from flat_pca.webui.services.view_selection import ViewSelection
from flat_pca.webui.settings import Settings, UiSettings
from flat_pca.webui.workspace import Workspace

# The synthetic stems in natural order.
STEMS = [f"s-{index:02d}" for index in range(SPECTRA_FILE_COUNT)]


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _register(client: TestClient, settings: Settings, paths: list[Path], run_id: str) -> str:
    """Register a succeeded transform run of the given spectra files (and its fit run)."""
    return register_shown_run(_workspace(client).database, settings, paths, run_id, "median")


def _shown(sidebar: str) -> list[str]:
    """Return the stems of the sidebar partial's read-only list of shown files."""
    return re.findall(r'<li data-stem="([^"]+)">', sidebar)


def _table(client: TestClient, **params: str) -> str:
    """Return the shown-file dialog's table fragment."""
    response = client.get("/sidebar/selection/files/table", params=params)
    assert response.status_code == 200
    return response.text


def _options(client: TestClient) -> list[str]:
    """Return the row keys of the shown-file dialog's table."""
    return re.findall(r'<tr data-dt-key="([^"]+)"', _table(client))


def _checked(table: str) -> list[str]:
    """Return the checked row keys of a table fragment."""
    return re.findall(
        r'value="([^"]+)" aria-label="Select [^"]+" data-dt-row-check checked', table
    )


def _dialog(client: TestClient) -> str:
    """Return the contents of the shown-file dialog."""
    response = client.get("/sidebar/selection/files/dialog")
    assert response.status_code == 200
    return response.text


def _save(client: TestClient, run: str, stems: list[str]) -> Mapping[str, str]:
    """Save the shown files as the dialog's "Select" does and return the response headers."""
    response = client.post(
        "/sidebar/selection/files", data={"run": run, "stems": json.dumps(stems)}
    )
    assert response.status_code == 200
    return response.headers


def _trigger(headers: Mapping[str, str]) -> object:
    """Return the events of a response's ``HX-Trigger`` header."""
    return json.loads(headers["HX-Trigger"])


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application on an empty workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


@pytest.fixture
def limited_client(settings: Settings) -> Iterator[TestClient]:
    """Run the application showing up to 2 files."""
    limited = settings.model_copy(update={"ui": UiSettings(explore_max_files=2)})
    with TestClient(create_app(limited)) as opened:
        yield opened


def test_every_page_loads_the_sidebar_choices(client: TestClient) -> None:
    """The sidebar loads the choices and reloads them when a fit or transform finishes."""
    html = client.get("/").text

    match = re.search(r'<div id="view-selection"[^>]*>', html, re.DOTALL)
    assert match is not None
    assert 'hx-get="/sidebar/selection"' in match.group(0)
    assert 'hx-trigger="load, fit-updated from:body, transform-updated from:body"' in match.group(0)


def test_every_page_has_the_empty_file_dialog(client: TestClient) -> None:
    """The dialog sits outside the layout; app.js fetches its contents when it opens."""
    html = client.get("/").text

    match = re.search(
        r'<dialog id="view-files-dialog"[^>]*>\s*<div class="dialog-content" data-dialog-content[^>]*></div>',
        html,
    )
    assert match is not None
    assert html.index('class="layout"') < html.index("busy-overlay") < match.start()


def test_sidebar_without_a_run_offers_nothing(client: TestClient) -> None:
    """Without a succeeded transform run, no file can be chosen."""
    sidebar = client.get("/sidebar/selection").text

    assert "<select" not in sidebar
    assert "data-view-files-open" not in sidebar
    assert "No files to choose" in sidebar
    dialog = _dialog(client)
    assert 'name="run" value=""' in dialog
    assert "No files to choose" in dialog
    assert "data-datatable" not in dialog
    assert 'class="button-primary" disabled>Select</button>' in dialog


def test_sidebar_shows_the_count_the_names_and_the_dialog_button(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The sidebar lists the chosen files read-only and opens the dialog to change them."""
    _register(client, settings, spectra_paths, "tr-1")

    sidebar = client.get("/sidebar/selection").text

    assert "0 / 20" in sidebar
    assert _shown(sidebar) == []
    assert "No files chosen" in sidebar
    assert "<input" not in sidebar
    assert "<form" not in sidebar
    button = re.search(r"<button[^>]*data-view-files-open[^>]*>(.*?)</button>", sidebar, re.DOTALL)
    assert button is not None
    assert 'aria-label="Choose shown files"' in button.group(0)
    assert ">open_in_new</span>" in button.group(1)

    choose_view(client, files=["s-10", "s-02"])
    sidebar = client.get("/sidebar/selection").text

    assert "2 / 20" in sidebar
    assert _shown(sidebar) == ["s-02", "s-10"]


def test_dialog_starts_from_the_chosen_files_of_the_newest_run(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The newest run is used by default; its table has its own id and the limit."""
    _register(client, settings, spectra_paths, "tr-1")
    _register(client, settings, spectra_paths[:6], "tr-2")
    choose_view(client, files=["s-03"])

    dialog = _dialog(client)

    assert 'name="run" value="tr-2"' in dialog
    assert 'id="view-files" class="dt-root" data-datatable hx-get="/sidebar/selection/files/table"' in dialog
    assert 'data-dt-max-selected="20"' in dialog
    assert """id="view-files-selection" name="stems" value='["s-03"]' form="view-files-form\"""" in dialog
    assert '<form id="view-files-form" hx-post="/sidebar/selection/files"' in dialog
    assert "data-view-choice" in dialog
    assert "data-dt-selected-count>1 / 20 selected<" in _table(client)
    assert "data-dialog-close" in dialog
    assert _options(client) == STEMS[:6]
    assert _checked(_table(client)) == ["s-03"]


def test_dialog_table_has_the_file_table_columns_and_filters(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Files missing from the catalog are rows with blank metadata; filters and sorting work."""
    _register(client, settings, spectra_paths, "tr-1")

    table = _table(client)

    headers = re.findall(r'<span class="dt-label">([^<]*)</span>', table)
    assert headers == ["file", "lot", "date", "yield_pct", "Steps", "(Step, Sequence)s", "rows"]
    types = re.findall(r'<span class="dt-type">([^<]*)</span>', table)
    assert types == ["str", "cat", "datetime[μs]", "f64", "i64", "i64", "i64"]
    assert 'name="view-files.search"' in table
    assert 'popovertarget="view-files-columns-menu"' in table
    assert 'name="view-files.null__lot"' in table
    row = re.search(r'<tr data-dt-key="s-00">(.*?)</tr>', table, re.DOTALL)
    assert row is not None
    assert re.findall(r"<td[^>]*>([^<]*)</td>", row.group(1))[1:] == [""] * 6

    assert re.findall(
        r'<tr data-dt-key="([^"]+)"',
        _table(client, **{"view-files.search": "s-1", "view-files.order": "desc"}),
    ) == ["s-11", "s-10"]
    assert client.get(
        "/sidebar/selection/files/table", params={"view-files.sort": "missing"}
    ).status_code == 400


def test_files_are_listed_in_natural_order(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """``t-2`` comes before ``t-10`` whatever the fit order."""
    renamed = []
    names = ("t-10", "t-2", "t-1", "t-21", "t-3", "t-100")
    for path, stem in zip(spectra_paths, names, strict=False):
        target = path.with_name(f"{stem}.parquet")
        path.rename(target)
        renamed.append(target)
    _register(client, settings, renamed, "tr-1")

    assert _options(client) == ["t-1", "t-2", "t-3", "t-10", "t-21", "t-100"]
    choose_view(client, files=["t-10", "t-2"])
    assert _shown(client.get("/sidebar/selection").text) == ["t-2", "t-10"]


def test_choosing_files_keeps_transform_targets_and_announces_the_change(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Unknown stems are ignored, and the change of the files is announced."""
    _register(client, settings, spectra_paths, "tr-1")

    headers = _save(client, "tr-1", ["s-10", "ghost", "s-02"])

    assert _trigger(headers) == {"view-selection-changed": {"changed": "files"}}
    assert _workspace(client).view_selection.stems == ["s-02", "s-10"]
    sidebar = client.get("/sidebar/selection").text
    assert _shown(sidebar) == ["s-02", "s-10"]
    assert "2 / 20" in sidebar


@pytest.mark.parametrize("stems", ["s-01", '{"stems": []}', "[1]"])
def test_a_selection_that_is_not_a_json_array_of_stems_is_rejected(
    client: TestClient, settings: Settings, spectra_paths: list[Path], stems: str
) -> None:
    """The selection is the table's hidden input; anything else is a bad request."""
    _register(client, settings, spectra_paths, "tr-1")
    choose_view(client, files=["s-03"])

    response = client.post("/sidebar/selection/files", data={"run": "tr-1", "stems": stems})

    assert response.status_code == 400
    assert _workspace(client).view_selection.stems == ["s-03"]


def test_choosing_another_run_clears_the_files(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Changing the run clears the files; choosing the same run keeps them."""
    _register(client, settings, spectra_paths, "tr-1")
    _register(client, settings, spectra_paths, "tr-2")
    choose_view(client, run="tr-2", files=["s-01"])

    choose_view(client, run="tr-2")
    assert _workspace(client).view_selection.stems == ["s-01"]

    response = client.post("/transform/show", data={"run": "tr-1"})

    assert _trigger(response.headers) == {"view-selection-changed": {"changed": "run"}}
    assert _workspace(client).view_selection.run_id == "tr-1"
    assert _workspace(client).view_selection.stems == []
    assert _shown(client.get("/sidebar/selection").text) == []
    assert _options(client) == STEMS


def test_unknown_run_is_rejected(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only a succeeded transform run can be shown."""
    _register(client, settings, spectra_paths, "tr-1")

    response = client.post("/transform/show", data={"run": "missing"})

    assert response.status_code == 400
    assert _workspace(client).view_selection.run_id is None


def test_choice_survives_new_runs_and_falls_back_when_gone(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A chosen run stays chosen after a newer one; a gone run falls back to the latest."""
    _register(client, settings, spectra_paths, "tr-1")
    choose_view(client, run="tr-1", files=["s-03"])
    _register(client, settings, spectra_paths[:6], "tr-2")

    assert _shown(client.get("/sidebar/selection").text) == ["s-03"]
    assert _options(client) == STEMS

    _workspace(client).view_selection.choose_run("gone")
    assert _shown(client.get("/sidebar/selection").text) == []
    assert _options(client) == STEMS[:6]


def test_files_beyond_the_limit_are_not_kept(
    limited_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """At ``ui.explore_max_files`` the other rows of the dialog cannot be checked."""
    _register(limited_client, settings, spectra_paths, "tr-1")

    choose_view(limited_client, files=["s-10", "s-01", "s-00"])

    assert _workspace(limited_client).view_selection.stems == ["s-00", "s-01"]
    assert "2 / 2" in limited_client.get("/sidebar/selection").text
    assert 'data-dt-max-selected="2"' in _dialog(limited_client)
    table = _table(limited_client)
    assert _checked(table) == ["s-00", "s-01"]
    assert 'value="s-02" aria-label="Select s-02" data-dt-row-check disabled' in table
    assert re.search(r"<p class=\"dt-limit\"[^>]*data-dt-part=\"limit\">", table) is not None


def test_adding_a_file_stops_at_the_limit(
    limited_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A clicked file is added once; at the limit it is refused with a message."""
    _register(limited_client, settings, spectra_paths, "tr-1")

    for stem in ("s-05", "s-05", "s-01"):
        response = limited_client.post(
            "/sidebar/selection/files/add", data={"run": "tr-1", "stem": stem}
        )
        assert response.status_code == 200
        assert response.json() == {"added": True}
    refused = limited_client.post(
        "/sidebar/selection/files/add", data={"run": "tr-1", "stem": "s-07"}
    )

    assert refused.status_code == 200
    assert refused.json()["added"] is False
    assert "Up to 2 shown files" in refused.json()["message"]
    assert _workspace(limited_client).view_selection.stems == ["s-05", "s-01"]


def test_adding_an_unknown_file_is_rejected(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only a transform target of the run in use can be added."""
    _register(client, settings, spectra_paths, "tr-1")

    response = client.post(
        "/sidebar/selection/files/add", data={"run": "tr-1", "stem": "ghost"}
    )

    assert response.status_code == 400


def test_a_figure_of_another_run_adds_nothing(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A click on a figure drawn before the shown run changed adds nothing."""
    _register(client, settings, spectra_paths, "tr-1")
    _register(client, settings, spectra_paths, "tr-2")

    response = client.post(
        "/sidebar/selection/files/add", data={"run": "tr-1", "stem": "s-01"}
    )

    assert response.status_code == 200
    assert response.json()["added"] is False
    assert "tr-2" in response.json()["message"]
    assert _workspace(client).view_selection.stems == []




def test_a_dialog_of_another_run_changes_nothing(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A selection made in a dialog opened before the run changed (in another tab) keeps the choice.

    The response announces a change of the run, so the browser also replaces
    the main part drawn with the previous run.
    """
    _register(client, settings, spectra_paths, "tr-1")
    _register(client, settings, spectra_paths, "tr-2")
    choose_view(client, run="tr-2", files=["s-02"])
    assert 'name="run" value="tr-2"' in _dialog(client)

    headers = _save(client, "tr-1", ["s-01"])

    assert _trigger(headers) == {"view-selection-changed": {"changed": "run"}}
    assert _workspace(client).view_selection.stems == ["s-02"]


def test_a_run_change_waits_for_an_addition(
    client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run chosen while a file is being added applies after the addition.

    The addition checks the run in use and adds to its files in one step,
    so the run change cannot slip in between and keep a file of the old run.
    """
    _register(client, settings, spectra_paths, "tr-1")
    _register(client, settings, spectra_paths, "tr-2")
    choose_view(client, run="tr-1")
    selection = _workspace(client).view_selection
    resolve = view_selection_routes.current_view_choice
    switch = threading.Thread(target=selection.choose_run, args=("tr-2",))

    def resolve_then_switch(workspace: Workspace) -> object:
        """Resolve the choice, then change the run from another thread."""
        choice = resolve(workspace)
        switch.start()
        switch.join(timeout=0.2)
        assert switch.is_alive(), "the run changed during the addition"
        return choice

    monkeypatch.setattr(
        view_selection_routes, "current_view_choice", resolve_then_switch
    )

    response = client.post(
        "/sidebar/selection/files/add", data={"run": "tr-1", "stem": "s-01"}
    )
    switch.join()

    assert response.json() == {"added": True}
    assert selection.run_id == "tr-2"
    assert selection.stems == []


def test_concurrent_additions_keep_each_other() -> None:
    """Additions racing from many threads all stay chosen, up to the limit."""
    selection = ViewSelection()
    selection.replace_stems(["gone"])
    options = [f"s-{index:02d}" for index in range(40)]
    barrier = threading.Barrier(len(options))

    def add(stem: str) -> bool:
        """Add one stem once every thread is ready."""
        barrier.wait()
        return selection.add_stem(stem, options, max_files=30)

    with ThreadPoolExecutor(len(options)) as pool:
        added = list(pool.map(add, options))

    assert sum(added) == 30
    assert sorted(selection.stems) == sorted(
        stem for stem, ok in zip(options, added, strict=True) if ok
    )
    assert "gone" not in selection.stems


def test_view_selection_clears_files_only_for_another_run() -> None:
    """``ViewSelection`` keeps the files while the run stays the same."""
    selection = ViewSelection()
    selection.choose_run("tr-1")
    selection.replace_stems(["a", "b"])

    selection.choose_run("tr-1")
    assert selection.stems == ["a", "b"]
    selection.choose_run("tr-2")
    assert selection.stems == []
    assert selection.run_id == "tr-2"


def test_fit_and_shown_files_have_distinct_labels(client: TestClient) -> None:
    """The data selection is labelled as the fit target, the sidebar as shown files."""
    status = client.get("/sidebar/status").text
    page = client.get("/").text

    assert "<dt>Fit target</dt>" in status
    assert ">Select</button>" in page
    assert "Fit target: 0 files" in page
    assert "Shown files" in client.get("/sidebar/selection").text
