"""Tests for the spectral exploration screen through FastAPI's ``TestClient``."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from spectra import SPECTRA_SHORT_FILE

from flat_pca.webui.app import create_app
from flat_pca.webui.jobs.fit_run import build_fit_config, run_fit
from flat_pca.webui.services.runs import insert_run, update_run
from flat_pca.webui.settings import Settings, UiSettings
from flat_pca.webui.workspace import FIT_JOB, Workspace

Wait = Callable[..., dict[str, object]]
STATISTICS = {"cumulative_explained_variance": 2, "alpha": 0.01}
FIT_RUN_ID = "fit-1"


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _embedded(html: str, element_id: str) -> dict[str, object]:
    """Return the JSON embedded in the page's ``<script id=element_id>``."""
    match = re.search(rf'id="{element_id}">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _decode(values: object) -> list[object]:
    """Decode a Plotly array, which may be a base64 typed array."""
    if isinstance(values, dict):
        array = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        if "shape" in values:
            array = array.reshape([int(size) for size in str(values["shape"]).split(",")])
        return array.tolist()
    return list(values)  # type: ignore[call-overload]


def _heatmap(html: str) -> dict[str, object]:
    """Return the heatmap trace embedded in the page."""
    return _embedded(html, "explore-heatmap-figure")["data"][0]  # type: ignore[index]


@pytest.fixture
def client(settings: Settings, wait_for: Wait) -> Iterator[TestClient]:
    """Run the application on a cataloged workspace."""
    with TestClient(create_app(settings)) as opened:
        workspace = _workspace(opened)
        run = wait_for(workspace.database, workspace.submit_catalog())
        assert run["status"] == "succeeded"
        yield opened


@pytest.fixture
def fit_run(client: TestClient, settings: Settings, spectra_paths: list[Path]) -> str:
    """Register a succeeded fit run of the synthetic spectra using ``sqrt``.

    The fixture's own files are too few to fit, so the run is executed
    directly on the synthetic spectra.

    Returns
    -------
    str
        The run's identifier.
    """
    config = build_fit_config(
        settings,
        [{"stem": path.stem, "path": str(path)} for path in spectra_paths],
        {
            "target_steps": [1, 2],
            "max_null_ratio": 0.1,
            "intensity_transform": "sqrt",
            "intensity_transform_scale": 2.0,
        },
        {
            "n_component": 3,
            "impute_strategy": "median",
            "impute_kmeans_n_clusters": None,
            "scaling_strategy": "none",
        },
        STATISTICS,
        STATISTICS,
    )
    run_dir = settings.runs_dir / FIT_RUN_ID
    run_dir.mkdir(parents=True)
    run_fit(json.loads(json.dumps(config)), run_dir)
    database = _workspace(client).database
    insert_run(database, FIT_RUN_ID, FIT_JOB, config, run_dir)
    update_run(database, FIT_RUN_ID, status="succeeded")
    return FIT_RUN_ID


def test_navigation_links_to_the_page(client: TestClient) -> None:
    """The sidebar enables the exploration screen, which loads Plotly.js."""
    html = client.get("/explore").text

    assert 'href="/explore" aria-current="page"' in html
    assert "vendor/plotly.min.js" in html


def test_raw_view_draws_the_first_file_and_segment(client: TestClient) -> None:
    """Without choices, the raw view shows the first file's first segment."""
    response = client.get("/explore")

    assert response.status_code == 200
    html = response.text
    assert '<option value="run-1" selected>' in html
    assert '<option value="1:1" selected>' in html
    assert 'data-heatmap-label="run-1"' in html
    assert "data-binned" not in html
    heatmap = _heatmap(html)
    assert _decode(heatmap["x"]) == [400.0, 401.0, 402.5]
    assert _decode(heatmap["y"]) == [0.0, 0.5, 1.0]
    layout = _embedded(html, "explore-heatmap-figure")["layout"]
    assert layout["yaxis"]["title"]["text"] == "StepTime"  # type: ignore[index]
    assert _embedded(html, "explore-axes") == {
        "wavelengths": [400.0, 401.0, 402.5],
        "step_times": [0.0, 0.5, 1.0],
    }
    assert (
        'data-trend-url="/explore/trend?view=raw&amp;file=run-1&amp;segment=1%3A1"' in html
    )


def test_raw_view_lists_the_selected_files(client: TestClient) -> None:
    """With a selection, only the selected files can be chosen."""
    client.post("/catalog/selection", data={"stems": ["run-2", "run-10"]})

    html = client.get("/explore", params={"file": ["run-10"], "segment": "2:2"}).text

    assert 'value="run-1"' not in html
    assert '<option value="run-10" selected>' in html
    assert '<option value="2:2" selected>' in html
    assert 'data-heatmap-label="run-10"' in html


def test_heatmap_is_binned_above_the_cell_limit(settings: Settings, wait_for: Wait) -> None:
    """Above ``ui.heatmap_max_cells`` the heatmap rows are averaged and marked."""
    limited = settings.model_copy(update={"ui": UiSettings(heatmap_max_cells=6)})
    with TestClient(create_app(limited)) as opened:
        workspace = _workspace(opened)
        wait_for(workspace.database, workspace.submit_catalog())
        html = opened.get("/explore").text

    assert "data-binned" in html
    assert "3 行を" in html
    # Two bins over StepTime 0..1: [0, 0.5) holds 0; [0.5, 1] holds 0.5 and 1.
    assert _decode(_heatmap(html)["y"]) == [0.0, 0.75]
    assert _embedded(html, "explore-axes")["step_times"] == [0.0, 0.5, 1.0]


def test_trend_overlays_files_at_the_nearest_point(client: TestClient) -> None:
    """Trends cut every chosen file at its nearest unbinned grid point."""
    response = client.get(
        "/explore/trend",
        params={
            "view": "raw",
            "file": ["run-1", "run-2"],
            "segment": "1:1",
            "wavelength": 401.2,
            "step_time": 0.6,
        },
    )

    assert response.status_code == 200
    trends = response.json()
    assert trends["wavelength"] == 401.0
    assert trends["step_time"] == 0.5
    by_step_time = trends["by_step_time"]["data"]
    assert [trace["name"] for trace in by_step_time] == ["run-1", "run-2"]
    assert _decode(by_step_time[0]["x"]) == [0.0, 0.5, 1.0]
    assert _decode(by_step_time[0]["y"]) == [0.0, 1.0, 2.0]
    by_wavelength = trends["by_wavelength"]["data"]
    assert _decode(by_wavelength[0]["x"]) == [400.0, 401.0, 402.5]
    assert _decode(by_wavelength[0]["y"]) == [0.5, 1.0, 1.5]


def test_preprocessed_view_shows_x_rows_and_the_transform(
    client: TestClient, fit_run: str
) -> None:
    """The preprocessed view reshapes X.npy rows and names the transform."""
    short = f"s-{SPECTRA_SHORT_FILE:02d}"

    html = client.get(
        "/explore", params={"view": "preprocessed", "file": [short], "segment": "2:1"}
    ).text

    assert f'<option value="{fit_run}" selected>' in html
    assert 'data-intensity-transform="sqrt"' in html
    assert "scale = 2" in html
    assert f'data-heatmap-label="{short}"' in html
    heatmap = _heatmap(html)
    assert _decode(heatmap["y"]) == [0.0, 1.0, 2.0]
    # The short file lacks the last Step 2 row, so its last heatmap row is blank.
    z = np.array(_decode(heatmap["z"]), dtype=np.float64)
    assert np.isnan(z[2]).all()
    assert not np.isnan(z[:2]).any()

    trends = client.get(
        "/explore/trend",
        params={
            "view": "preprocessed",
            "file": ["s-00", short],
            "segment": "2:1",
            "wavelength": 400,
            "step_time": 2,
        },
    ).json()
    names = [trace["name"] for trace in trends["by_step_time"]["data"]]
    assert names == ["s-00", short]


def test_component_view_shows_one_component(client: TestClient, fit_run: str) -> None:
    """The component view reshapes the chosen component with a diverging scale."""
    html = client.get("/explore", params={"view": "component", "k": "2"}).text

    assert 'data-heatmap-label="PC2"' in html
    assert 'name="k" min="1" max="3" value="2"' in html
    assert 'name="file"' not in html
    assert "data-intensity-transform" not in html
    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "coefficient"
    assert coloraxis["cmin"] == -coloraxis["cmax"]

    trends = client.get(
        "/explore/trend",
        params={"view": "component", "k": "2", "wavelength": 400, "step_time": 0},
    ).json()
    assert [trace["name"] for trace in trends["by_step_time"]["data"]] == ["PC2"]


def test_run_views_without_a_run_show_a_message(client: TestClient) -> None:
    """Without a succeeded fit run, run-based views explain why nothing is drawn."""
    html = client.get("/explore", params={"view": "preprocessed"}).text

    assert "data-explore-error" in html
    assert "成功した fit run がありません" in html
    assert "data-explore " not in html
    response = client.get(
        "/explore/trend", params={"view": "component", "wavelength": 0, "step_time": 0}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [{"view": "residual"}, {"segment": "1"}, {"segment": "a:b"}, {"run": "missing"}],
)
def test_invalid_parameters_are_rejected(client: TestClient, params: dict[str, str]) -> None:
    """Unknown views, malformed segments, and unknown runs return 400."""
    assert client.get("/explore", params=params).status_code == 400


def _add_runs(client: TestClient, settings: Settings, count: int, status: str) -> None:
    """Register ``count`` fit runs newer than the existing ones, without artifacts."""
    database = _workspace(client).database
    for index in range(count):
        run_id = f"extra-{index:03d}"
        insert_run(database, run_id, FIT_JOB, {}, settings.runs_dir / run_id)
        update_run(database, run_id, status=status)


def test_succeeded_run_is_found_behind_many_failed_runs(
    client: TestClient, settings: Settings, fit_run: str
) -> None:
    """Newer failed runs do not push the succeeded run out of the choices."""
    _add_runs(client, settings, 101, "failed")

    html = client.get("/explore", params={"view": "component"}).text

    assert f'<option value="{fit_run}" selected>' in html
    assert 'data-heatmap-label="PC1"' in html


def test_old_succeeded_run_can_be_requested(
    client: TestClient, settings: Settings, fit_run: str
) -> None:
    """A succeeded run beyond the listed ones is still shown when requested."""
    _add_runs(client, settings, 101, "succeeded")

    html = client.get("/explore", params={"view": "component", "run": fit_run}).text

    assert f'<option value="{fit_run}" selected>' in html
    assert 'data-heatmap-label="PC1"' in html
