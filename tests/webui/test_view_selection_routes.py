"""Tests for the sidebar's run and file choices and the transform screen."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run
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
    """Register a succeeded fit run of the given spectra files."""
    return register_fit_run(_workspace(client).database, settings, paths, run_id, "median")


def _checked(sidebar: str) -> list[str]:
    """Return the checked stems of the sidebar partial."""
    return re.findall(r'name="file" value="([^"]+)" checked', sidebar)


def _options(sidebar: str) -> list[str]:
    """Return the stems of the sidebar partial's file choices."""
    return re.findall(r'<li data-stem="([^"]+)">', sidebar)


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
    """The sidebar loads the choices and reloads them when a fit finishes."""
    html = client.get("/").text

    match = re.search(r'<div id="view-selection"[^>]*>', html, re.DOTALL)
    assert match is not None
    assert 'hx-get="/sidebar/selection"' in match.group(0)
    assert 'hx-trigger="load, fit-updated from:body"' in match.group(0)


def test_sidebar_without_a_run_offers_nothing(client: TestClient) -> None:
    """Without a succeeded fit run, neither a run nor a file can be chosen."""
    sidebar = client.get("/sidebar/selection").text

    assert "data-no-run" in sidebar
    assert 'name="run"' not in sidebar
    assert 'name="file"' not in sidebar
    assert "選べるファイルがありません" in sidebar


def test_sidebar_lists_runs_and_transform_targets(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The newest run is used by default; its files are listed in natural order."""
    _register(client, settings, spectra_paths, "fit-1")
    _register(client, settings, spectra_paths[:6], "fit-2")

    sidebar = client.get("/sidebar/selection").text

    assert re.findall(r'<option value="([^"]+)"', sidebar) == ["fit-2", "fit-1"]
    assert '<option value="fit-2" selected>fit-2（' in sidebar
    assert _options(sidebar) == STEMS[:6]
    assert _checked(sidebar) == []
    assert "0 / 20" in sidebar
    assert "data-view-file-search" in sidebar
    assert 'hx-post="/sidebar/selection/run"' in sidebar
    assert 'hx-post="/sidebar/selection/files"' in sidebar


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
    _register(client, settings, renamed, "fit-1")

    assert _options(client.get("/sidebar/selection").text) == [
        "t-1", "t-2", "t-3", "t-10", "t-21", "t-100"
    ]


def test_choosing_files_keeps_transform_targets_and_reloads(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Unknown stems are ignored, and the page is reloaded."""
    _register(client, settings, spectra_paths, "fit-1")

    response = client.post(
        "/sidebar/selection/files",
        data={"run": "fit-1", "file": ["s-10", "ghost", "s-02"]},
    )

    assert response.status_code == 200
    assert response.headers["HX-Refresh"] == "true"
    assert _workspace(client).view_selection.stems == ["s-02", "s-10"]
    sidebar = client.get("/sidebar/selection").text
    assert _checked(sidebar) == ["s-02", "s-10"]
    assert "2 / 20" in sidebar


def test_choosing_another_run_clears_the_files(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Changing the run clears the files; choosing the same run keeps them."""
    _register(client, settings, spectra_paths, "fit-1")
    _register(client, settings, spectra_paths, "fit-2")
    choose_view(client, run="fit-2", files=["s-01"])

    choose_view(client, run="fit-2")
    assert _workspace(client).view_selection.stems == ["s-01"]

    response = client.post("/sidebar/selection/run", data={"run": "fit-1"})

    assert response.headers["HX-Refresh"] == "true"
    assert _workspace(client).view_selection.run_id == "fit-1"
    assert _workspace(client).view_selection.stems == []
    assert '<option value="fit-1" selected>' in client.get("/sidebar/selection").text


def test_unknown_run_is_rejected(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only a succeeded fit run can be chosen."""
    _register(client, settings, spectra_paths, "fit-1")

    response = client.post("/sidebar/selection/run", data={"run": "missing"})

    assert response.status_code == 400
    assert _workspace(client).view_selection.run_id is None


def test_choice_survives_new_runs_and_falls_back_when_gone(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A chosen run stays chosen after a newer fit; a gone run falls back to the latest."""
    _register(client, settings, spectra_paths, "fit-1")
    choose_view(client, run="fit-1", files=["s-03"])
    _register(client, settings, spectra_paths[:6], "fit-2")

    sidebar = client.get("/sidebar/selection").text
    assert '<option value="fit-1" selected>' in sidebar
    assert _checked(sidebar) == ["s-03"]

    _workspace(client).view_selection.choose_run("gone")
    sidebar = client.get("/sidebar/selection").text
    assert '<option value="fit-2" selected>' in sidebar
    assert _options(sidebar) == STEMS[:6]


def test_files_beyond_the_limit_are_not_kept(
    limited_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """At ``ui.explore_max_files`` the other files cannot be checked."""
    _register(limited_client, settings, spectra_paths, "fit-1")

    choose_view(limited_client, files=["s-10", "s-01", "s-00"])

    assert _workspace(limited_client).view_selection.stems == ["s-00", "s-01"]
    sidebar = limited_client.get("/sidebar/selection").text
    assert _checked(sidebar) == ["s-00", "s-01"]
    assert 'name="file" value="s-02" disabled' in sidebar
    assert "data-view-file-limit" in sidebar
    assert "上限の 2 件" in sidebar


def test_adding_a_file_stops_at_the_limit(
    limited_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A clicked file is added once; at the limit it is refused with a message."""
    _register(limited_client, settings, spectra_paths, "fit-1")

    for stem in ("s-05", "s-05", "s-01"):
        response = limited_client.post(
            "/sidebar/selection/files/add", data={"run": "fit-1", "stem": stem}
        )
        assert response.status_code == 200
        assert response.json() == {"added": True}
    refused = limited_client.post(
        "/sidebar/selection/files/add", data={"run": "fit-1", "stem": "s-07"}
    )

    assert refused.status_code == 200
    assert refused.json()["added"] is False
    assert "2 件まで" in refused.json()["message"]
    assert _workspace(limited_client).view_selection.stems == ["s-05", "s-01"]


def test_adding_an_unknown_file_is_rejected(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only a transform target of the run in use can be added."""
    _register(client, settings, spectra_paths, "fit-1")

    response = client.post(
        "/sidebar/selection/files/add", data={"run": "fit-1", "stem": "ghost"}
    )

    assert response.status_code == 400


def test_a_figure_of_another_run_adds_nothing(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A click on a figure drawn before the sidebar's run changed adds nothing."""
    _register(client, settings, spectra_paths, "fit-1")
    _register(client, settings, spectra_paths, "fit-2")

    response = client.post(
        "/sidebar/selection/files/add", data={"run": "fit-1", "stem": "s-01"}
    )

    assert response.status_code == 200
    assert response.json()["added"] is False
    assert "fit-2" in response.json()["message"]
    assert _workspace(client).view_selection.stems == []


def test_a_sidebar_of_another_run_changes_nothing(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Checkboxes drawn before the run changed (in another tab) keep the choice."""
    _register(client, settings, spectra_paths, "fit-1")
    _register(client, settings, spectra_paths, "fit-2")
    choose_view(client, run="fit-2", files=["s-02"])
    assert 'name="run" value="fit-2"' in client.get("/sidebar/selection").text

    response = client.post(
        "/sidebar/selection/files", data={"run": "fit-1", "file": ["s-01"]}
    )

    assert response.headers["HX-Refresh"] == "true"
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
    _register(client, settings, spectra_paths, "fit-1")
    _register(client, settings, spectra_paths, "fit-2")
    choose_view(client, run="fit-1")
    selection = _workspace(client).view_selection
    resolve = view_selection_routes.current_view_choice
    switch = threading.Thread(target=selection.choose_run, args=("fit-2",))

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
        "/sidebar/selection/files/add", data={"run": "fit-1", "stem": "s-01"}
    )
    switch.join()

    assert response.json() == {"added": True}
    assert selection.run_id == "fit-2"
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
    selection.choose_run("fit-1")
    selection.replace_stems(["a", "b"])

    selection.choose_run("fit-1")
    assert selection.stems == ["a", "b"]
    selection.choose_run("fit-2")
    assert selection.stems == []
    assert selection.run_id == "fit-2"


def test_transform_page_follows_the_fit_data(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The transform page comes after the preprocessing page and fixes its checkbox."""
    html = client.get("/transform").text

    assert 'href="/transform" aria-current="page"' in html
    assert html.index('href="/fit"') < html.index('href="/transform"') < html.index(
        'href="/model"'
    )
    assert re.search(
        r'<input type="checkbox" name="use_same_data_for_fit" checked disabled', html
    )
    assert "use same data for fit" in html
    assert "成功した fit run がありません" in html

    _register(client, settings, spectra_paths, "fit-1")

    html = client.get("/transform").text
    assert f'data-transform-targets="{SPECTRA_FILE_COUNT}"' in html
    assert f"run fit-1 の transform 対象：{SPECTRA_FILE_COUNT} ファイル" in html


def test_fit_and_shown_files_have_distinct_labels(client: TestClient) -> None:
    """The data selection is labelled as the fit target, the sidebar as shown files."""
    status = client.get("/sidebar/status").text
    page = client.get("/").text

    assert "<dt>fit 対象</dt>" in status
    assert "このファイル集合を fit 対象にする" in page
    assert "fit 対象：0 ファイル" in page
    assert "表示ファイル" in client.get("/sidebar/selection").text
