"""Tests for the preprocessing and PCA screen through FastAPI's ``TestClient``."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flat_pca.webui.app import create_app
from flat_pca.webui.jobs.progress import write_progress
from flat_pca.webui.services.catalog_query import selection_ranges
from flat_pca.webui.services.fit_form import default_form_values
from flat_pca.webui.services.runs import get_run, insert_run, latest_run, update_run
from flat_pca.webui.settings import JobsSettings, Settings
from flat_pca.webui.workspace import CATALOG_JOB, FIT_JOB, Workspace

Wait = Callable[..., dict[str, object]]
STEMS = ["run-1", "run-2", "run-10"]


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _errors(html: str) -> dict[str, str]:
    """Return the field errors rendered in ``html``."""
    return dict(re.findall(r'data-error-for="([^"]+)">([^<]*)<', html))


def _cataloged(client: TestClient, wait_for: Wait) -> TestClient:
    """Run one catalog update and select the fixture files."""
    workspace = _workspace(client)
    run = wait_for(workspace.database, workspace.submit_catalog())
    assert run["status"] == "succeeded"
    client.post("/catalog/selection", data={"stems": json.dumps(STEMS)})
    return client


@pytest.fixture
def client(settings: Settings, wait_for: Wait) -> Iterator[TestClient]:
    """Run the application on a cataloged workspace with three files selected."""
    with TestClient(create_app(settings)) as opened:
        yield _cataloged(opened, wait_for)


@pytest.fixture
def form(client: TestClient) -> dict[str, object]:
    """Return the default form values for the selected files."""
    database = _workspace(client).database
    return default_form_values(selection_ranges(database, STEMS))


def _post(client: TestClient, path: str, values: dict[str, object]) -> str:
    """Post form values, leaving out unchecked checkboxes."""
    data = {name: value for name, value in values.items() if value is not False}
    response = client.post(path, data=data)
    assert response.status_code == 200
    return response.text


def test_page_renders_the_form_from_the_catalog(client: TestClient) -> None:
    """The page lists the catalog's steps and ranges and the estimate."""
    response = client.get("/fit")

    assert response.status_code == 200
    html = response.text
    assert 'href="/fit" aria-current="page"' in html
    assert 'name="target_steps" value="1" checked' in html
    assert 'name="target_steps" value="2" checked' in html
    assert 'name="wavelength_range_lower" value="400"' in html
    assert 'name="t_normalization_range_upper" value="3"' in html
    assert 'data-n-features="21"' in html
    assert "Q の UCL" in html
    assert 'name="outlier_strategy"' not in html
    assert "No fit has run yet." in html
    assert 'hx-get="/fit/runs"' in html


def test_fit_groups_are_collapsible(client: TestClient) -> None:
    """Each settings group, the run status, and the run list are open accordions."""
    html = client.get("/fit").text

    for group in (
        "target",
        "target-steps",
        "preprocess",
        "pca",
        "t2-q",
        "estimate",
        "status",
        "runs",
    ):
        assert f'<details class="card" data-group="{group}" open>' in html
    assert "<fieldset" not in html


def _active_run(client: TestClient, tmp_path: Path, progress: tuple[int, int] | None) -> str:
    """Register a running fit run, with ``(done, total)`` progress if given."""
    database = _workspace(client).database
    run_dir = tmp_path / "fit-active"
    run_dir.mkdir()
    insert_run(database, "fit-active", FIT_JOB, {}, run_dir)
    update_run(database, "fit-active", status="running")
    if progress is not None:
        write_progress(run_dir, "fit", *progress)
    return "fit-active"


def test_active_run_shows_progress_for_the_overlay(
    client: TestClient, tmp_path: Path
) -> None:
    """An active run's panel carries the overlay's elapsed time, cancel URL, and ETA."""
    run_id = _active_run(client, tmp_path, (1, 4))

    html = client.get("/fit").text

    panel = re.search(r"<div class=\"status-panel\" data-job-status[^>]*>", html)
    assert panel is not None
    assert "data-job-active" in panel.group()
    assert f'data-run-id="{run_id}"' in panel.group()
    assert re.search(r'data-elapsed-s="\d+\.\d"', panel.group())
    assert f'data-cancel-url="/runs/{run_id}/cancel"' in panel.group()
    assert "Stage fit: 1 / 4" in html
    assert re.search(r"Remaining \(all stages\): about \d+ s", html)


def test_active_run_without_done_stages_is_still_estimating(
    client: TestClient, tmp_path: Path
) -> None:
    """Until a stage is done the remaining time reads as being calculated."""
    run_id = _active_run(client, tmp_path, (0, 4))

    html = client.get(f"/fit/runs/{run_id}/status?polling=true").text

    assert "Remaining (all stages): estimating" in html
    assert "hx-trigger=\"every 1s\"" in html


def test_finished_run_leaves_no_overlay_data(
    client: TestClient, tmp_path: Path
) -> None:
    """A finished run's panel no longer asks for the overlay."""
    run_id = _active_run(client, tmp_path, None)
    update_run(_workspace(client).database, run_id, status="cancelled")

    response = client.get(f"/fit/runs/{run_id}/status?polling=true")

    assert response.headers["HX-Trigger"] == "fit-updated"
    assert "data-job-active" not in response.text
    assert "data-job-progress" not in response.text
    assert "Cancelled." in response.text


def test_estimate_inside_the_form_sets_its_own_target(client: TestClient) -> None:
    """The estimate does not inherit the form's swap target."""
    html = client.get("/fit").text

    form_tag = re.search(r'<form id="fit-form"[^>]*>', html)
    estimate_tag = re.search(r'<div id="fit-estimate"[^>]*>', html)
    assert form_tag is not None and estimate_tag is not None
    assert 'hx-swap="outerHTML"' in form_tag.group()
    assert 'hx-post="/fit/estimate"' in estimate_tag.group()
    assert 'hx-target="this"' in estimate_tag.group()
    assert 'hx-swap="innerHTML"' in estimate_tag.group()


def test_page_without_selection_disables_submit(settings: Settings) -> None:
    """Without selected files the submit button is disabled."""
    with TestClient(create_app(settings)) as opened:
        html = opened.get("/fit").text

    assert re.search(r'<button type="submit" form="fit-form" disabled>Run fit</button>', html)


def test_estimate_follows_the_form(client: TestClient, form: dict[str, object]) -> None:
    """The estimate partial reflects the selected steps and strides."""
    html = _post(
        client,
        "/fit/estimate",
        {**form, "target_steps": ["1"], "w_downsampling_stride": "2"},
    )

    assert 'data-n-features="6"' in html
    assert "<form" not in html


def test_invalid_values_return_the_form_with_errors(
    client: TestClient, form: dict[str, object]
) -> None:
    """Validation errors are shown next to their fields and nothing is queued."""
    html = _post(
        client,
        "/fit",
        {
            **form,
            "wavelength_range_enabled": "1",
            "wavelength_range_lower": "402",
            "wavelength_range_upper": "401",
            "mahalanobis_alpha": "2",
            "n_component": "abc",
        },
    )

    errors = _errors(html)
    assert set(errors) == {"wavelength_range", "mahalanobis_alpha", "n_component"}
    assert 'id="fit-form"' in html
    assert 'name="wavelength_range_lower" value="402"' in html
    assert "hx-swap-oob" not in html
    assert latest_run(_workspace(client).database, FIT_JOB) is None


def test_submit_queues_a_fit_run(
    client: TestClient, form: dict[str, object], wait_for: Wait
) -> None:
    """A valid form queues a fit job and swaps in its status out of band."""
    response = client.post(
        "/fit",
        data={name: value for name, value in form.items() if value is not False},
    )

    assert response.status_code == 200
    assert response.headers["HX-Trigger"] == "fit-started"
    assert _errors(response.text) == {}
    assert '<div id="fit-run-status" hx-swap-oob="true">' in response.text
    workspace = _workspace(client)
    run = latest_run(workspace.database, FIT_JOB)
    assert run is not None
    config = json.loads(str(run["config_json"]))
    assert [file["stem"] for file in config["files"]] == STEMS
    assert config["files"][0]["metadata"] == {
        "lot": "A",
        "date": "2026-01-02T03:04:05",
        "yield_pct": 91.5,
    }
    assert config["preprocess"]["target_steps"] == [1, 2]
    assert config["pca"]["n_component"] is None
    assert config["artifact_dtype"] == "float32"

    finished = wait_for(workspace.database, str(run["run_id"]))
    polled = client.get(f"/fit/runs/{run['run_id']}/status?polling=true")
    assert polled.status_code == 200
    assert polled.headers["HX-Trigger"] == "fit-updated"
    assert f'data-run-status="{finished["status"]}"' in polled.text
    assert "hx-trigger" not in polled.text


def test_polled_success_shows_icon_next_to_submit(
    client: TestClient, form: dict[str, object], wait_for: Wait
) -> None:
    """A poll that finds the run succeeded swaps a check icon next to the button."""
    form_html = _post(client, "/fit", form)
    assert '<span id="fit-result" class="fit-result" hx-swap-oob="true"></span>' in form_html
    workspace = _workspace(client)
    run = latest_run(workspace.database, FIT_JOB)
    assert run is not None
    # The fixture files are too few to fit; only the finished status matters.
    wait_for(workspace.database, str(run["run_id"]))
    update_run(workspace.database, str(run["run_id"]), status="succeeded")

    polled = client.get(f"/fit/runs/{run['run_id']}/status?polling=true").text
    reopened = client.get(f"/fit/runs/{run['run_id']}/status").text

    assert '<span id="fit-result" class="fit-result" hx-swap-oob="true">' in polled
    assert "check_circle" in polled
    assert "fit-result" not in reopened


def test_failed_run_shows_its_error(
    client: TestClient, form: dict[str, object], wait_for: Wait
) -> None:
    """A run whose data reject the settings shows the job's error message."""
    _post(
        client,
        "/fit",
        {
            **form,
            "wavelength_range_enabled": "1",
            "wavelength_range_lower": "403",
            "wavelength_range_upper": "404",
        },
    )
    workspace = _workspace(client)
    run = latest_run(workspace.database, FIT_JOB)
    assert run is not None
    finished = wait_for(workspace.database, str(run["run_id"]))

    html = client.get(f"/fit/runs/{run['run_id']}/status").text

    assert finished["status"] == "failed"
    assert 'data-run-status="failed"' in html
    polled = client.get(f"/fit/runs/{run['run_id']}/status?polling=true").text
    assert "check_circle" not in polled
    assert "ValueError" in html
    assert "wavelength" in html


@pytest.fixture
def small_limit_client(settings: Settings, wait_for: Wait) -> Iterator[TestClient]:
    """Run the application with a tiny ``jobs.memory_warn_gb``."""
    limited = settings.model_copy(update={"jobs": JobsSettings(memory_warn_gb=1e-9)})
    with TestClient(create_app(limited)) as opened:
        yield _cataloged(opened, wait_for)


def test_memory_above_the_limit_needs_confirmation(
    small_limit_client: TestClient, form: dict[str, object]
) -> None:
    """Above ``jobs.memory_warn_gb`` the job runs only when confirmed."""
    html = _post(small_limit_client, "/fit", form)

    assert "data-memory-warning" in html
    assert set(_errors(html)) == {"confirm_memory"}
    database = _workspace(small_limit_client).database
    assert latest_run(database, FIT_JOB) is None

    _post(small_limit_client, "/fit", {**form, "confirm_memory": "1"})

    assert latest_run(database, FIT_JOB) is not None


def test_run_list_and_reopened_run(
    client: TestClient, form: dict[str, object], wait_for: Wait
) -> None:
    """Past runs are listed and reopen with their settings."""
    _post(client, "/fit", {**form, "n_component": "2", "target_steps": ["2"]})
    workspace = _workspace(client)
    run = latest_run(workspace.database, FIT_JOB)
    assert run is not None
    run_id = str(run["run_id"])
    wait_for(workspace.database, run_id)

    listed = client.get("/fit/runs").text
    reopened = client.get(f"/fit?run={run_id}").text

    assert f'href="/fit?run={run_id}"' in listed
    assert 'name="n_component" value="2"' in reopened
    assert 'name="target_steps" value="1" >' in reopened
    assert 'name="target_steps" value="2" checked' in reopened
    assert run_id in reopened


def test_unknown_runs_are_not_found(client: TestClient) -> None:
    """Unknown runs and runs of other kinds return 404."""
    catalog_run = latest_run(_workspace(client).database, CATALOG_JOB)
    assert catalog_run is not None

    assert client.get("/fit?run=missing").status_code == 404
    assert client.get(f"/fit?run={catalog_run['run_id']}").status_code == 404
    assert client.get("/fit/runs/missing/status").status_code == 404
    assert get_run(_workspace(client).database, "missing") is None


def test_run_button_leads_the_status_card(client: TestClient) -> None:
    """The run button sits in the run status card and submits the form from there."""
    html = client.get("/fit").text

    form = html.split('<form id="fit-form"', 1)[1].split("</form>", 1)[0]
    status = html.split('data-group="status"', 1)[1].split("</details>", 1)[0]
    assert '<button type="submit"' not in form
    assert '<button type="submit" form="fit-form" >Run fit</button>' in status
    assert status.index("Run fit") < status.index('id="fit-run-status"')
