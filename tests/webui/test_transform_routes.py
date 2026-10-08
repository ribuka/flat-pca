"""Tests for the transform screen, its job, and the display of transform runs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run, register_transform_run
from spectra import SPECTRA_FILE_COUNT, write_spectra
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.routes.view_selection import current_view_choice
from flat_pca.webui.services.monitoring import MonitoringRequest, resolve_monitoring
from flat_pca.webui.services.runs import insert_run, update_run
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import FIT_JOB, Workspace

Wait = Callable[..., dict[str, object]]
# Catalog stems of the Web UI fixture data.
CATALOG_STEMS = ["run-1", "run-2", "run-10"]


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _register_fit(client: TestClient, settings: Settings, paths: list[Path]) -> None:
    """Register the succeeded fit run ``fit-1`` of the given spectra files."""
    register_fit_run(_workspace(client).database, settings, paths, "fit-1", "median")


def _register_transform(
    client: TestClient, settings: Settings, tmp_path: Path, run_id: str = "tr-1"
) -> list[Path]:
    """Register a transform run of three spectra on fewer wavelengths with ``fit-1``."""
    targets = write_spectra(
        tmp_path / run_id, n_files=3, wavelengths=(400.0, 401.0, 402.5)
    )
    register_transform_run(
        _workspace(client).database, settings, "fit-1", targets, run_id
    )
    return targets


def _submit(client: TestClient, stems: list[str]) -> object:
    """Post the transform form with the chosen stems."""
    return client.post("/transform", data={"stems": json.dumps(stems)})


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application on an empty workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


@pytest.fixture
def cataloged_client(client: TestClient, wait_for: Wait) -> TestClient:
    """Return a client after one successful catalog run."""
    workspace = _workspace(client)
    assert (
        wait_for(workspace.database, workspace.submit_catalog())["status"]
        == "succeeded"
    )
    return client


def test_page_without_a_fit_run(client: TestClient) -> None:
    """The page follows the preprocessing page; nothing can be transformed yet."""
    html = client.get("/transform").text

    assert 'href="/transform" aria-current="page"' in html
    assert (
        html.index('href="/fit"')
        < html.index('href="/transform"')
        < html.index('href="/model"')
    )
    assert re.search(
        r'<input type="checkbox" name="use_same_data_for_fit" data-use-same-data checked>',
        html,
    )
    assert "No succeeded fit run" in html
    assert re.search(r'<button id="run-transform"[^>]*disabled', html)
    assert "No transform has run yet." in html


def test_page_with_a_fit_run_uses_the_fit_data(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Checked by default: the fit data are shown and the target choice is hidden."""
    _register_fit(client, settings, spectra_paths)

    html = client.get("/transform").text

    assert 'data-transform-model="fit-1"' in html
    assert f'data-transform-shown="{SPECTRA_FILE_COUNT}"' in html
    assert re.search(r"data-use-same-data checked", html)
    assert re.search(r"<details[^>]*data-transform-targets open hidden>", html)
    # Checking the box has nothing to switch back to.
    form = re.search(r'<form id="use-same-data-form"[^>]*>', html)
    assert form is not None
    assert "hx-post" not in form.group()
    assert 'hx-get="/catalog/files"' in html
    assert re.search(
        r'<button id="run-transform"[^>]*hx-include="#selected-stems"', html
    )
    assert 'hx-get="/transform/runs"' in html


def test_submit_requires_targets_and_a_fit_run(
    cataloged_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Without a fit run or a chosen file, the form reports it and queues nothing."""
    response = _submit(cataloged_client, CATALOG_STEMS)
    assert "data-transform-error" in response.text
    assert "No succeeded fit run" in response.text

    _register_fit(cataloged_client, settings, spectra_paths)
    response = _submit(cataloged_client, ["not-cataloged"])
    assert "Choose the files to transform." in response.text
    assert "HX-Trigger" not in response.headers
    assert cataloged_client.get("/transform/runs").text.count("data-run-id") == 0


def test_submit_rejects_invalid_stems(client: TestClient) -> None:
    """The chosen stems must be a JSON array of strings."""
    response = client.post("/transform", data={"stems": "{"})

    assert response.status_code == 400


def test_submit_runs_a_transform_job(
    cataloged_client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    wait_for: Wait,
) -> None:
    """A transform job runs with the sidebar's fit run on the chosen catalog files."""
    _register_fit(cataloged_client, settings, spectra_paths)
    workspace = _workspace(cataloged_client)

    response = _submit(cataloged_client, ["run-2", "run-1"])

    assert response.headers["HX-Trigger"] == "transform-started"
    assert '<div id="transform-run-status" hx-swap-oob="innerHTML">' in response.text
    assert workspace.transform_selection.stems == ["run-1", "run-2"]
    run_id = re.search(r'data-run-id="([^"]+)"', response.text)
    assert run_id is not None
    run = wait_for(workspace.database, run_id.group(1))
    assert run["status"] == "succeeded", run["error"]
    assert run["kind"] == "transform"
    assert run["n_files"] == 2
    assert json.loads(str(run["config_json"]))["fit_run_id"] == "fit-1"

    polled = cataloged_client.get(
        f"/transform/runs/{run['run_id']}/status?polling=true"
    )
    assert polled.headers["HX-Trigger"] == "transform-updated"
    assert "data-run-summary" in polled.text
    runs = cataloged_client.get("/transform/runs").text
    assert f'<tr data-run-id="{run["run_id"]}">' in runs
    assert '<td class="mono">fit-1</td>' in runs
    assert "data-show-run" in runs
    # The page keeps the chosen targets.
    assert 'value=\'["run-1", "run-2"]\'' in cataloged_client.get("/transform").text


def test_status_of_another_kind_is_not_found(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only transform runs have a transform status."""
    _register_fit(client, settings, spectra_paths)

    assert client.get("/transform/runs/fit-1/status").status_code == 404
    assert client.get("/transform/runs/missing/status").status_code == 404


def test_sidebar_lists_transform_runs_under_their_fit_run(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Each fit run is followed by its transform runs, newest first."""
    _register_fit(client, settings, spectra_paths)
    register_fit_run(
        _workspace(client).database, settings, spectra_paths[:6], "fit-2", "median"
    )
    _register_transform(client, settings, tmp_path, "tr-1")
    _register_transform(client, settings, tmp_path, "tr-2")

    sidebar = client.get("/sidebar/selection").text

    assert re.findall(r'<option value="([^"]+)"', sidebar) == [
        "fit-2",
        "fit-1",
        "tr-2",
        "tr-1",
    ]
    assert (
        '<option value="tr-1" data-run-kind="transform">└ transform tr-1（' in sidebar
    )
    assert "transform-updated from:body" in client.get("/").text


def test_chosen_transform_run_shows_its_targets(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """Choosing a transform run shows its files on every display screen."""
    _register_fit(client, settings, spectra_paths)
    targets = _register_transform(client, settings, tmp_path)
    stems = [path.stem for path in targets]

    choose_view(client, run="tr-1", files=stems[:2])

    choice = current_view_choice(_workspace(client))
    assert (choice.run_id, choice.fit_run_id) == ("tr-1", "fit-1")
    assert choice.file_options == stems
    assert choice.files == stems[:2]
    for path in (
        "/explore?view=preprocessed",
        "/explore?view=reconstruction",
        "/scores",
        "/monitoring",
        "/model",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "Cannot read the run's artifacts" not in response.text, path
    monitoring = client.get("/monitoring").text
    assert all(stem in monitoring for stem in stems)
    assert "s-11" not in monitoring

    html = client.get("/transform").text
    assert 'data-transform-model="fit-1"' in html
    assert re.search(r"<details[^>]*data-transform-targets open >", html)
    form = re.search(r'<form id="use-same-data-form"[^>]*>', html)
    assert form is not None
    assert 'hx-post="/sidebar/selection/run"' in form.group()
    assert '<input type="hidden" name="run" value="fit-1">' in html
    assert not re.search(r"data-use-same-data checked", html)
    assert "data-shown-run" in client.get("/transform/runs").text


def test_transform_run_uses_the_fit_run_control_limits(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The T² and Q control limits of a transform run are its fit run's."""
    _register_fit(client, settings, spectra_paths)
    _register_transform(client, settings, tmp_path)

    workspace = _workspace(client)
    runs = current_view_choice(workspace).runs
    limits = {}
    for run in ("fit-1", "tr-1"):
        shown = resolve_monitoring(
            workspace.cache, runs, MonitoringRequest(run=run), None
        )
        assert shown.points is not None, shown.error
        limits[run] = (shown.points.t2_ucl, shown.points.q_ucl)
    assert limits["tr-1"] == limits["fit-1"]


def test_transform_run_of_an_old_fit_run_can_be_chosen(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """A transform run is offered even when its fit run is older than the listed ones."""
    _register_fit(client, settings, spectra_paths)
    targets = _register_transform(client, settings, tmp_path)
    database = _workspace(client).database
    for index in range(101):
        run_id = f"extra-{index:03d}"
        insert_run(database, run_id, FIT_JOB, {}, settings.runs_dir / run_id)
        update_run(database, run_id, status="succeeded")

    choose_view(client, run="tr-1")

    choice = current_view_choice(_workspace(client))
    assert (choice.run_id, choice.fit_run_id) == ("tr-1", "fit-1")
    assert choice.file_options == [path.stem for path in targets]
    run_ids = [str(run["run_id"]) for run in choice.runs]
    assert run_ids[-2:] == ["fit-1", "tr-1"]
    assert run_ids[0] == "extra-100"
