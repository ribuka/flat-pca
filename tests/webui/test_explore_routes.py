"""Tests for the spectral exploration screen through FastAPI's ``TestClient``."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run
from spectra import SPECTRA_SHORT_FILE

from flat_pca.webui.app import create_app
from flat_pca.webui.services.display_cache import DisplayCache
from flat_pca.webui.services.runs import insert_run, update_run
from flat_pca.webui.settings import Settings, UiSettings
from flat_pca.webui.workspace import FIT_JOB, Workspace

Wait = Callable[..., dict[str, object]]
FIT_RUN_ID = "fit-1"


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _clear_cache(client: TestClient) -> None:
    """Replace the workspace's display cache with an empty one."""
    workspace = _workspace(client)
    workspace.cache = DisplayCache(workspace.settings.ui.explore_max_files)


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
    """Register a succeeded z-score fit run imputing missing values with the median.

    Returns
    -------
    str
        The run's identifier.
    """
    return register_fit_run(
        _workspace(client).database, settings, spectra_paths, FIT_RUN_ID, "median"
    )


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


def test_run_views_without_a_run_show_a_message(client: TestClient) -> None:
    """Without a succeeded fit run, run-based views explain why nothing is drawn."""
    html = client.get("/explore", params={"view": "preprocessed"}).text

    assert "data-explore-error" in html
    assert "成功した fit run がありません" in html
    assert "data-explore " not in html
    response = client.get(
        "/explore/trend", params={"view": "preprocessed", "wavelength": 0, "step_time": 0}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"view": "unknown"},
        {"view": "component"},
        {"segment": "1"},
        {"segment": "a:b"},
        {"run": "missing"},
    ],
)
def test_invalid_parameters_are_rejected(client: TestClient, params: dict[str, str]) -> None:
    """Unknown views (including the moved component view), malformed segments,
    and unknown runs return 400."""
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

    html = client.get("/explore", params={"view": "preprocessed"}).text

    assert f'<option value="{fit_run}" selected>' in html
    assert 'data-heatmap-label="s-00"' in html


def test_old_succeeded_run_can_be_requested(
    client: TestClient, settings: Settings, fit_run: str
) -> None:
    """A succeeded run beyond the listed ones is still shown when requested."""
    _add_runs(client, settings, 101, "succeeded")

    html = client.get("/explore", params={"view": "preprocessed", "run": fit_run}).text

    assert f'<option value="{fit_run}" selected>' in html
    assert 'data-heatmap-label="s-00"' in html


def _trend_values(
    client: TestClient, params: dict[str, object]
) -> dict[str, np.ndarray]:
    """Return the wavelength-trend values of a view keyed by trace name.

    Parameters
    ----------
    client : TestClient
        Client of the application.
    params : dict[str, object]
        Query parameters selecting the view, without the point.

    Returns
    -------
    dict[str, np.ndarray]
        Values over wavelength at ``StepTime`` 2 of every trace.
    """
    response = client.get(
        "/explore/trend", params={**params, "wavelength": 400, "step_time": 2}
    )
    assert response.status_code == 200
    return {
        trace["name"]: np.array(_decode(trace["y"]), dtype=np.float64)
        for trace in response.json()["by_wavelength"]["data"]
    }


def test_reconstruction_view_has_no_missing_cells_and_overlays_x(
    client: TestClient, fit_run: str
) -> None:
    """The imputed file is reconstructed everywhere; trends overlay X.npy."""
    short = f"s-{SPECTRA_SHORT_FILE:02d}"
    params = {"view": "reconstruction", "file": [short], "segment": "2:1", "k": "2"}

    html = client.get("/explore", params=params).text

    assert f'data-heatmap-label="{short}（累積再構成）"' in html
    assert 'name="k" min="1" max="3" value="2"' in html
    assert 'data-intensity-transform="sqrt"' in html
    assert "先頭 1..2 成分の累積再構成" in html
    z = np.array(_decode(_heatmap(html)["z"]), dtype=np.float64)
    assert z.shape == (3, 5)
    assert np.isfinite(z).all()
    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "intensity"

    lines = _trend_values(client, params)
    assert list(lines) == [f"{short}（累積再構成）", f"{short}（前処理済み）"]
    assert np.isfinite(lines[f"{short}（累積再構成）"]).all()
    # X.npy lacks the short file's last Step 2 row.
    assert np.isnan(lines[f"{short}（前処理済み）"]).all()


def test_residual_view_has_no_nan_for_an_imputed_file(
    client: TestClient, fit_run: str
) -> None:
    """Residuals of imputed rows are finite and drawn on a diverging scale."""
    short = f"s-{SPECTRA_SHORT_FILE:02d}"
    params = {"view": "residual", "file": [short], "segment": "2:1", "k": "1"}

    html = client.get("/explore", params=params).text

    assert f'data-heatmap-label="{short}"' in html
    z = np.array(_decode(_heatmap(html)["z"]), dtype=np.float64)
    assert np.isfinite(z).all()
    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "residual"
    assert coloraxis["cmin"] == -coloraxis["cmax"]
    assert np.isfinite(_trend_values(client, params)[short]).all()


def test_residual_plus_reconstruction_is_the_preprocessed_row(
    client: TestClient, fit_run: str
) -> None:
    """Where X.npy has no missing value, residual + reconstruction = X."""
    common = {"file": ["s-00"], "segment": "2:1", "k": "2"}

    reconstruction = _trend_values(client, {**common, "view": "reconstruction"})
    residual = _trend_values(client, {**common, "view": "residual"})["s-00"]

    np.testing.assert_allclose(
        reconstruction["s-00（累積再構成）"] + residual,
        reconstruction["s-00（前処理済み）"],
    )


def test_contribution_view_is_the_reconstruction_increment(
    client: TestClient, fit_run: str
) -> None:
    """The k-th contribution is the change of the reconstruction from k-1 to k."""
    common = {"file": ["s-00"], "segment": "2:1"}

    html = client.get("/explore", params={**common, "view": "contribution", "k": "2"}).text
    contribution = _trend_values(client, {**common, "view": "contribution", "k": "2"})
    first = _trend_values(client, {**common, "view": "reconstruction", "k": "1"})
    second = _trend_values(client, {**common, "view": "reconstruction", "k": "2"})

    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "contribution"
    assert coloraxis["cmin"] == -coloraxis["cmax"]
    assert "第 2 成分のみの寄与" in html
    np.testing.assert_allclose(
        contribution["s-00"],
        second["s-00（累積再構成）"] - first["s-00（累積再構成）"],
    )


def test_drop_strategy_excludes_files_with_missing_values(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Files dropped by ``impute_strategy="drop"`` are listed, not reconstructed."""
    run_id = register_fit_run(
        _workspace(client).database, settings, spectra_paths, "fit-drop", "drop"
    )
    short = f"s-{SPECTRA_SHORT_FILE:02d}"

    html = client.get(
        "/explore",
        params={"view": "residual", "run": run_id, "file": ["s-00", short]},
    ).text
    alone = client.get(
        "/explore", params={"view": "residual", "run": run_id, "file": [short]}
    ).text

    assert 'data-heatmap-label="s-00"' in html
    assert "data-dropped" in html
    assert short in html.split("data-dropped", 1)[1].split("</p>", 1)[0]
    assert "data-explore-error" in alone
    assert "補完方法 drop で除外される" in alone


def _trend_url(html: str, wavelength: float, step_time: float) -> str:
    """Return the trend URL that the page's JavaScript requests at one point."""
    match = re.search(r'data-trend-url="([^"]*)"', html)
    assert match is not None
    point = urlencode({"wavelength": wavelength, "step_time": step_time})
    return f"{match.group(1).replace('&amp;', '&')}&{point}"


def test_trend_of_a_shown_view_reuses_its_matrices(
    client: TestClient, fit_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trends of a drawn page are cut from its matrices without resolving the view."""
    short = f"s-{SPECTRA_SHORT_FILE:02d}"
    params = {"view": "contribution", "file": ["s-00", short], "segment": "2:1", "k": "2"}
    point = {"wavelength": 400, "step_time": 2}
    expected = client.get(
        "/explore/trend", params={**params, "run": fit_run, **point}
    ).json()
    assert len(expected["by_step_time"]["data"]) == 2
    _clear_cache(client)
    html = client.get("/explore", params=params).text

    def fail(*args: object, **kwargs: object) -> None:
        """Fail if the view is resolved or rows are prepared again."""
        raise AssertionError("the shown view was resolved again")

    monkeypatch.setattr("flat_pca.webui.routes.explore.resolve_explore", fail)
    monkeypatch.setattr("flat_pca.webui.services.display_cache.prepare_rows", fail)

    for step_time in (0, 1, 2):
        assert client.get(_trend_url(html, 401, step_time)).status_code == 200
    assert client.get(_trend_url(html, **point)).json() == expected


def test_trend_is_resolved_again_after_eviction(client: TestClient, fit_run: str) -> None:
    """Trends are the same after the kept matrices are evicted."""
    html = client.get(
        "/explore",
        params={"view": "residual", "file": ["s-00"], "segment": "2:1", "k": "1"},
    ).text
    url = _trend_url(html, wavelength=400, step_time=2)
    expected = client.get(url).json()
    assert expected["step_time"] == 2

    _clear_cache(client)

    assert client.get(url).json() == expected


@pytest.fixture
def limited_client(settings: Settings, wait_for: Wait) -> Iterator[TestClient]:
    """Run the application on a cataloged workspace showing up to 2 files."""
    limited = settings.model_copy(update={"ui": UiSettings(explore_max_files=2)})
    with TestClient(create_app(limited)) as opened:
        workspace = _workspace(opened)
        run = wait_for(workspace.database, workspace.submit_catalog())
        assert run["status"] == "succeeded"
        yield opened


def test_files_beyond_the_limit_are_not_shown(limited_client: TestClient) -> None:
    """Above ``ui.explore_max_files`` the leading files are shown and the rest listed."""
    html = limited_client.get(
        "/explore", params={"file": ["run-1", "run-2", "run-10"]}
    ).text

    assert '<option value="run-1" selected>' in html
    assert '<option value="run-2" selected>' in html
    assert '<option value="run-10" selected>' not in html
    omitted = html.split("data-omitted", 1)[1].split("</p>", 1)[0]
    assert "2 件まで" in omitted
    assert "（1 件）：run-10" in omitted
    assert "file=run-1&amp;file=run-2&amp;segment" in html


def test_files_within_the_limit_list_nothing_omitted(limited_client: TestClient) -> None:
    """Up to ``ui.explore_max_files`` files, nothing is reported as omitted."""
    html = limited_client.get("/explore", params={"file": ["run-1", "run-2"]}).text

    assert "data-omitted" not in html
    assert "一度に表示できるのは 2 件までです" in html


def test_trend_at_the_file_limit_prepares_no_row_again(
    limited_client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the limit of files chosen, resolving a trend again reuses the prepared rows."""
    workspace = _workspace(limited_client)
    run_id = register_fit_run(
        workspace.database, settings, spectra_paths, FIT_RUN_ID, "median"
    )
    assert workspace.cache.prepared_row_entries == 2
    html = limited_client.get(
        "/explore",
        params={"view": "residual", "run": run_id, "file": ["s-00", "s-01"], "k": "1"},
    ).text
    assert "data-omitted" not in html

    def fail(*args: object, **kwargs: object) -> None:
        """Fail if rows are prepared again."""
        raise AssertionError("rows were prepared again")

    # Evict the kept matrices, so the trend resolves the view again.
    monkeypatch.setattr(workspace.cache, "shown_matrices", lambda key: None)
    monkeypatch.setattr("flat_pca.webui.services.display_cache.prepare_rows", fail)

    response = limited_client.get(_trend_url(html, wavelength=400, step_time=1))

    assert response.status_code == 200
    assert len(response.json()["by_step_time"]["data"]) == 2
