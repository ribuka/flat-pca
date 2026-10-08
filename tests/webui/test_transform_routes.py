"""Tests for the transform screen, its job, and the display of transform runs."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run, register_transform_run
from spectra import SPECTRA_FILE_COUNT, write_spectra
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.routes.view_selection import current_model_run, current_view_choice
from flat_pca.webui.run_layout import SCORES_FILE
from flat_pca.webui.services.monitoring import MonitoringRequest, resolve_monitoring
from flat_pca.webui.services.runs import insert_run, list_runs, update_run
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import TRANSFORM_JOB, Workspace

Wait = Callable[..., dict[str, object]]
# Catalog stems of the Web UI fixture data.
CATALOG_STEMS = ["run-1", "run-2", "run-10"]
CHANGED_RUN = '{"view-selection-changed": {"changed": "run"}}'
CHANGED_MODEL = '{"view-selection-changed": {"changed": "model"}}'


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _register_fit(
    client: TestClient, settings: Settings, paths: list[Path], run_id: str = "fit-1"
) -> None:
    """Register a succeeded fit run of the given spectra files."""
    register_fit_run(_workspace(client).database, settings, paths, run_id, "median")


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


def _submit(
    client: TestClient,
    stems: list[str],
    model: str | None = None,
    use_same_data: bool | None = None,
) -> object:
    """Post "Run transform" with the chosen stems and the page's settings.

    The model and the checkbox default to those a page drawn now shows.
    """
    page = _workspace(client)
    if model is None and (shown := current_model_run(page)) is not None:
        model = str(shown["run_id"])
    if use_same_data is None:
        use_same_data = page.transform_settings.use_same_data
    data = {"stems": json.dumps(stems)}
    if model is not None:
        data["model"] = model
    if use_same_data:
        data["use_same_data"] = "true"
    return client.post("/transform", data=data)


def _settings(
    client: TestClient, model: str, use_same_data: bool, stems: list[str] | None = None
) -> object:
    """Post the model and the checkbox as the transform screen's form does."""
    data = {"model": model, "stems": json.dumps(stems or [])}
    if use_same_data:
        data["use_same_data"] = "true"
    return client.post("/transform/settings", data=data)


def _selected_stems(html: str) -> list[str]:
    """Return the chosen stems of the page's ``#selected-stems``."""
    match = re.search(r"id=\"selected-stems\" name=\"stems\" value='([^']*)'", html)
    assert match is not None
    return json.loads(match.group(1))


def _transform_run_ids(client: TestClient) -> list[str]:
    """Return the identifiers of every transform run, newest first."""
    runs = list_runs(_workspace(client).database, TRANSFORM_JOB, limit=None)
    return [str(run["run_id"]) for run in runs]


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
    assert "data-no-model" in html
    assert "data-use-same-data" not in html
    assert "<details class=\"card\" data-group=\"targets\" data-transform-targets open>" in html
    assert re.search(r'<button id="run-transform"[^>]*disabled', html)
    assert "No transform has run yet." in html


def test_page_with_a_fit_run_locks_the_fit_targets(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Checked by default: the table and the button show, with the fit targets locked."""
    _register_fit(client, settings, spectra_paths)

    html = client.get("/transform").text

    assert re.search(r'<option value="fit-1" selected>fit-1（', html)
    assert re.search(r"data-use-same-data checked", html)
    assert "<details class=\"card\" data-group=\"targets\" data-transform-targets open>" in html
    assert re.search(r'<div id="file-table"[^>]*data-locked>', html)
    assert f'data-fit-targets="{SPECTRA_FILE_COUNT}"' in html
    assert _selected_stems(html) == [path.stem for path in spectra_paths]
    assert 'hx-post="/transform/settings"' in html
    assert re.search(
        r'<button id="run-transform"[^>]*hx-include="#selected-stems, #transform-settings-form"[^>]*>', html
    )
    assert not re.search(r'<button id="run-transform"[^>]*disabled', html)
    assert 'hx-get="/transform/runs"' in html


def test_fit_alone_transforms_nothing(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """A fit run holds no scores, so the display screens wait for a transform run."""
    _register_fit(client, settings, spectra_paths)

    assert not (settings.runs_dir / "fit-1" / SCORES_FILE).exists()
    assert _transform_run_ids(client) == []
    assert "No succeeded transform run" in client.get("/scores").text
    assert "No transform run yet." in client.get("/transform/runs").text


def test_settings_choose_the_model_and_the_kind_of_targets(
    cataloged_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The model and the checkbox are the screen's state; unchecked targets survive."""
    _register_fit(cataloged_client, settings, spectra_paths)
    _register_fit(cataloged_client, settings, spectra_paths[:6], "fit-2")
    assert re.search(r'<option value="fit-2" selected>', cataloged_client.get("/transform").text)

    response = _settings(cataloged_client, "fit-1", use_same_data=False)

    assert response.status_code == 200
    assert response.headers["HX-Trigger"] == CHANGED_MODEL
    html = cataloged_client.get("/transform").text
    assert re.search(r'<option value="fit-1" selected>', html)
    assert not re.search(r"data-use-same-data checked", html)
    assert "data-locked" not in html
    assert "data-fit-targets" not in html
    assert _selected_stems(html) == []

    # Checking again saves the targets chosen while unchecked.
    _settings(cataloged_client, "fit-2", use_same_data=True, stems=["run-2"])
    html = cataloged_client.get("/transform").text
    assert _selected_stems(html) == [path.stem for path in spectra_paths[:6]]
    _settings(cataloged_client, "fit-2", use_same_data=False, stems=["ignored"])
    assert _selected_stems(cataloged_client.get("/transform").text) == ["run-2"]


def test_settings_reject_an_unknown_model(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only a succeeded fit run can be the model."""
    _register_fit(client, settings, spectra_paths)

    assert _settings(client, "missing", use_same_data=True).status_code == 400
    response = client.post("/transform/settings", data={"model": "fit-1", "stems": "{"})
    assert response.status_code == 400


def test_submit_requires_targets_and_a_fit_run(
    cataloged_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Without a fit run or a chosen file, the form reports it and queues nothing."""
    response = _submit(cataloged_client, CATALOG_STEMS)
    assert "data-transform-error" in response.text
    assert "No succeeded fit run" in response.text

    _register_fit(cataloged_client, settings, spectra_paths)
    _settings(cataloged_client, "fit-1", use_same_data=False)
    response = _submit(cataloged_client, ["not-cataloged"])
    assert "Choose the files to transform." in response.text
    assert "HX-Trigger" not in response.headers
    assert _transform_run_ids(cataloged_client) == []


def test_submit_rejects_invalid_stems(client: TestClient) -> None:
    """The chosen stems must be a JSON array of strings."""
    response = client.post("/transform", data={"stems": "{"})

    assert response.status_code == 400


def test_submit_with_the_same_data_transforms_the_fit_targets_once(
    client: TestClient, settings: Settings, spectra_paths: list[Path], wait_for: Wait
) -> None:
    """Checked, the fit targets are transformed; pressing again shows that run."""
    _register_fit(client, settings, spectra_paths)
    workspace = _workspace(client)

    response = _submit(client, ["ignored"])

    assert response.headers["HX-Trigger"] == "transform-started"
    run_id = re.search(r'data-run-id="([^"]+)"', response.text)
    assert run_id is not None
    assert workspace.view_selection.run_id == run_id.group(1)
    run = wait_for(workspace.database, run_id.group(1))
    assert run["status"] == "succeeded", run["error"]
    config = json.loads(str(run["config_json"]))
    assert config["fit_run_id"] == "fit-1"
    assert [file["stem"] for file in config["files"]] == [path.stem for path in spectra_paths]
    choice = current_view_choice(workspace)
    assert (choice.run_id, choice.fit_run_id) == (run["run_id"], "fit-1")
    assert len(choice.file_options) == SPECTRA_FILE_COUNT

    choose_view(client, files=choice.file_options[:1])
    response = _submit(client, [])

    assert response.headers["HX-Trigger"] == CHANGED_RUN
    assert f'data-transform-reused="{run["run_id"]}"' in response.text
    assert _transform_run_ids(client) == [run["run_id"]]
    # Showing the run that is already shown keeps the shown files.
    assert current_view_choice(workspace).files == choice.file_options[:1]

    # A target file changed since the run is transformed again.
    changed = spectra_paths[0]
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    response = _submit(client, [])

    assert response.headers["HX-Trigger"] == "transform-started"
    rerun = re.search(r'data-run-id="([^"]+)"', response.text)
    assert rerun is not None
    assert wait_for(workspace.database, rerun.group(1))["status"] == "succeeded"
    assert _transform_run_ids(client) == [rerun.group(1), run["run_id"]]


def test_submit_uses_the_settings_the_page_was_drawn_with(
    client: TestClient, settings: Settings, spectra_paths: list[Path], wait_for: Wait
) -> None:
    """A model chosen in another tab does not change what an older page transforms."""
    _register_fit(client, settings, spectra_paths)
    _register_fit(client, settings, spectra_paths[:6], "fit-2")
    # Another tab chooses fit-2 and its own targets after this page showed fit-1.
    _settings(client, "fit-2", use_same_data=True)

    response = _submit(client, [], model="fit-1", use_same_data=True)

    run_id = re.search(r'data-run-id="([^"]+)"', response.text)
    assert run_id is not None
    run = wait_for(_workspace(client).database, run_id.group(1))
    assert json.loads(str(run["config_json"]))["fit_run_id"] == "fit-1"
    assert run["n_files"] == SPECTRA_FILE_COUNT
    gone = _submit(client, [], model="missing", use_same_data=True)
    assert "Fit run missing is no longer available. Reload the page." in gone.text
    assert "HX-Trigger" not in gone.headers


def test_submit_runs_a_transform_job_of_the_chosen_files(
    cataloged_client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    wait_for: Wait,
) -> None:
    """Unchecked, a job transforms the chosen catalog files with the chosen model."""
    _register_fit(cataloged_client, settings, spectra_paths)
    _settings(cataloged_client, "fit-1", use_same_data=False)
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
    assert f'data-transform-shown="{run["run_id"]}"' in runs
    assert "data-shown-run" in runs
    # The page keeps the chosen targets.
    assert _selected_stems(cataloged_client.get("/transform").text) == ["run-1", "run-2"]

    # The same files chosen again (in another order) are not transformed again.
    response = _submit(cataloged_client, ["run-1", "run-2"])
    assert response.headers["HX-Trigger"] == CHANGED_RUN
    assert f'data-transform-reused="{run["run_id"]}"' in response.text
    assert _transform_run_ids(cataloged_client) == [run["run_id"]]


def test_status_of_another_kind_is_not_found(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Only transform runs have a transform status."""
    _register_fit(client, settings, spectra_paths)

    assert client.get("/transform/runs/fit-1/status").status_code == 404
    assert client.get("/transform/runs/missing/status").status_code == 404


def test_show_chooses_the_transform_run_of_the_display_screens(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The newest transform run is shown until another one is shown from the run list."""
    _register_fit(client, settings, spectra_paths)
    _register_transform(client, settings, tmp_path, "tr-1")
    _register_transform(client, settings, tmp_path, "tr-2")
    runs = client.get("/transform/runs").text
    assert 'data-transform-shown="tr-2"' in runs
    assert re.search(r'hx-post="/transform/show"[^>]*>\s*<input type="hidden" name="run" value="tr-1">', runs)

    response = client.post("/transform/show", data={"run": "tr-1"})

    assert response.status_code == 200
    assert response.headers["HX-Trigger"] == CHANGED_RUN
    assert current_view_choice(_workspace(client)).run_id == "tr-1"
    assert 'data-transform-shown="tr-1"' in client.get("/transform/runs").text
    assert client.post("/transform/show", data={"run": "fit-1"}).status_code == 400
    assert client.post("/transform/show", data={"run": "missing"}).status_code == 400


def test_sidebar_has_no_run_choice(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The sidebar chooses the shown files only; the run is chosen on the transform screen."""
    _register_fit(client, settings, spectra_paths)
    targets = _register_transform(client, settings, tmp_path)

    sidebar = client.get("/sidebar/selection").text

    assert "<select" not in sidebar
    assert "/sidebar/selection/run" not in sidebar
    assert re.findall(r'<li data-stem="([^"]+)"', sidebar) == [path.stem for path in targets]
    assert client.post("/sidebar/selection/run", data={"run": "tr-1"}).status_code == 404
    assert "transform-updated from:body" in client.get("/").text


def test_chosen_transform_run_shows_its_targets(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The shown transform run's files are shown on every display screen."""
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
    assert "data-shown-run" in client.get("/transform/runs").text


def test_transform_run_uses_the_fit_run_control_limits(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """The T² and Q control limits of a transform run come from its fit run's model."""
    _register_fit(client, settings, spectra_paths)
    register_transform_run(
        _workspace(client).database, settings, "fit-1", spectra_paths, "tr-fit"
    )
    _register_transform(client, settings, tmp_path)

    workspace = _workspace(client)
    runs = current_view_choice(workspace).runs
    limits = {}
    for run in ("tr-fit", "tr-1"):
        shown = resolve_monitoring(
            workspace.cache, runs, MonitoringRequest(run=run), None, None
        )
        assert shown.points is not None, shown.error
        limits[run] = (shown.points.t2_ucl, shown.points.q_ucl)
    assert limits["tr-1"] == limits["tr-fit"]


def test_old_transform_run_can_be_shown(
    client: TestClient, settings: Settings, spectra_paths: list[Path], tmp_path: Path
) -> None:
    """A transform run older than the listed ones can still be shown."""
    _register_fit(client, settings, spectra_paths)
    targets = _register_transform(client, settings, tmp_path)
    choose_view(client, run="tr-1")
    database = _workspace(client).database
    for index in range(101):
        run_id = f"extra-{index:03d}"
        config = {"fit_run_id": "fit-1", "fit_run_dir": str(settings.runs_dir / "fit-1")}
        insert_run(database, run_id, TRANSFORM_JOB, config, settings.runs_dir / run_id)
        update_run(database, run_id, status="succeeded")

    choice = current_view_choice(_workspace(client))
    assert (choice.run_id, choice.fit_run_id) == ("tr-1", "fit-1")
    assert choice.file_options == [path.stem for path in targets]
    run_ids = [str(run["run_id"]) for run in choice.runs]
    assert run_ids[0] == "extra-100"
    assert run_ids[-1] == "tr-1"
