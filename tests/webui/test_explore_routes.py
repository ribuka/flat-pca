"""Tests for the spectral exploration screen through FastAPI's ``TestClient``."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run
from spectra import SPECTRA_SHORT_FILE, SPECTRA_WAVELENGTHS
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.services.display_cache import DisplayCache
from flat_pca.webui.services.runs import insert_run, update_run
from flat_pca.webui.settings import Settings, UiSettings
from flat_pca.webui.workspace import FIT_JOB, Workspace

Wait = Callable[..., dict[str, object]]
FIT_RUN_ID = "fit-1"
SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"


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


def _raw_values(path: Path, step: int) -> np.ndarray:
    """Return one Step of a synthetic spectra file as a time × wavelength matrix."""
    frame = pl.read_parquet(path).filter(pl.col("Step") == step).sort("Time")
    columns = [f"{wavelength:.1f}nm" for wavelength in SPECTRA_WAVELENGTHS]
    return frame.select(columns).to_numpy()


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


def test_page_has_no_run_or_file_choice(client: TestClient, fit_run: str) -> None:
    """The run and the files are chosen in the sidebar, not on the page."""
    html = client.get("/explore").text

    assert 'name="run"' not in html
    assert 'name="file"' not in html
    assert f"run {fit_run} の表示ファイル：s-00。" in html


def test_raw_view_draws_the_first_transform_target(
    client: TestClient, fit_run: str, spectra_paths: list[Path]
) -> None:
    """Without chosen files, the raw view shows the run's first file and segment."""
    response = client.get("/explore")

    assert response.status_code == 200
    html = response.text
    assert '<option value="1:1" selected>' in html
    assert 'value="2:1" >' in html
    assert 'data-heatmap-label="s-00"' in html
    assert "data-binned" not in html
    heatmap = _heatmap(html)
    assert _decode(heatmap["x"]) == list(SPECTRA_WAVELENGTHS)
    assert _decode(heatmap["y"]) == [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_allclose(
        np.array(_decode(heatmap["z"]), dtype=np.float64), _raw_values(spectra_paths[0], 1)
    )
    layout = _embedded(html, "explore-heatmap-figure")["layout"]
    assert layout["yaxis"]["title"]["text"] == "StepTime"  # type: ignore[index]
    assert "title" not in layout
    assert '<option value="s-00" selected>s-00</option>' in html
    assert _embedded(html, "explore-axes") == {
        "wavelengths": list(SPECTRA_WAVELENGTHS),
        "step_times": [0.0, 1.0, 2.0, 3.0],
    }
    assert (
        f'data-trend-url="/explore/trend?view=raw&amp;run={fit_run}&amp;file=s-00'
        '&amp;heatmap_file=s-00&amp;segment=1%3A1"' in html
    )


def test_raw_view_shows_the_chosen_files_in_natural_order(
    client: TestClient, fit_run: str
) -> None:
    """The chosen files are shown in natural order; the first is the heatmap."""
    choose_view(client, files=["s-10", "s-02"])

    html = client.get("/explore", params={"segment": "2:1"}).text

    assert '<option value="2:1" selected>' in html
    assert 'data-heatmap-label="s-02"' in html
    assert "表示ファイル：s-02、s-10。" in html
    assert "file=s-02&amp;file=s-10&amp;heatmap_file=s-02&amp;segment=2%3A1" in html


@pytest.mark.parametrize(
    ("view", "label"),
    [
        ("raw", "s-10"),
        ("preprocessed", "s-10"),
        ("contribution", "s-10"),
        ("reconstruction", "s-10 (reconstruction)"),
        ("residual", "s-10"),
        ("q_contribution", "s-10"),
    ],
)
def test_heatmap_draws_the_chosen_file(
    client: TestClient, fit_run: str, view: str, label: str
) -> None:
    """Every file view draws the chosen heatmap file; the trends lead with it."""
    choose_view(client, files=["s-10", "s-02"])

    html = client.get("/explore", params={"view": view, "heatmap_file": "s-10"}).text

    assert f'data-heatmap-label="{label}"' in html
    assert '<select name="heatmap_file" form="explore-form">' in html
    assert '<option value="s-02" >s-02</option>' in html
    assert '<option value="s-10" selected>s-10</option>' in html
    assert "title" not in _embedded(html, "explore-heatmap-figure")["layout"]
    assert "heatmap_file=s-10" in html
    trends = client.get(_trend_url(html, wavelength=400, step_time=0)).json()
    names = [trace["name"] for trace in trends["by_step_time"]["data"]]
    assert names[0] == label
    assert any(name.startswith("s-02") for name in names)


def test_unchosen_heatmap_file_falls_back_to_the_first(
    client: TestClient, fit_run: str
) -> None:
    """A heatmap file outside the sidebar's files draws the first file instead."""
    choose_view(client, files=["s-10", "s-02"])

    html = client.get("/explore", params={"heatmap_file": "s-00"}).text

    assert 'data-heatmap-label="s-02"' in html
    assert '<option value="s-02" selected>s-02</option>' in html
    assert 'value="s-00"' not in html
    assert "heatmap_file=s-02" in html


def test_raw_view_without_a_run_asks_for_a_fit(client: TestClient) -> None:
    """The raw view chooses from the transform targets, so it needs a fit run."""
    html = client.get("/explore").text

    assert "data-explore-error" in html
    assert "Run a fit on Preprocess / PCA first." in html
    assert "data-explore " not in html


def test_heatmap_is_binned_above_the_cell_limit(
    settings: Settings, spectra_paths: list[Path]
) -> None:
    """Above ``ui.heatmap_max_cells`` the heatmap rows are averaged and marked."""
    limited = settings.model_copy(update={"ui": UiSettings(heatmap_max_cells=10)})
    with TestClient(create_app(limited)) as opened:
        register_fit_run(
            _workspace(opened).database, limited, spectra_paths, FIT_RUN_ID, "median"
        )
        html = opened.get("/explore").text

    assert "data-binned" in html
    assert "the 4 rows along time" in html
    # Two bins over StepTime 0..3: [0, 1.5) holds 0 and 1; [1.5, 3] holds 2 and 3.
    assert _decode(_heatmap(html)["y"]) == [0.5, 2.5]
    assert _embedded(html, "explore-axes")["step_times"] == [0.0, 1.0, 2.0, 3.0]


def test_trend_overlays_files_at_the_nearest_point(
    client: TestClient, fit_run: str, spectra_paths: list[Path]
) -> None:
    """Trends cut every chosen file at its nearest unbinned grid point."""
    response = client.get(
        "/explore/trend",
        params={
            "view": "raw",
            "run": fit_run,
            "file": ["s-00", "s-01"],
            "segment": "1:1",
            "wavelength": 401.2,
            "step_time": 0.6,
        },
    )

    assert response.status_code == 200
    trends = response.json()
    assert trends["wavelength"] == 401.0
    assert trends["step_time"] == 1.0
    by_step_time = trends["by_step_time"]["data"]
    assert [trace["name"] for trace in by_step_time] == ["s-00", "s-01"]
    assert _decode(by_step_time[0]["x"]) == [0.0, 1.0, 2.0, 3.0]
    raw = _raw_values(spectra_paths[0], 1)
    np.testing.assert_allclose(_decode(by_step_time[0]["y"]), raw[:, 1])
    by_wavelength = trends["by_wavelength"]["data"]
    assert _decode(by_wavelength[0]["x"]) == list(SPECTRA_WAVELENGTHS)
    np.testing.assert_allclose(_decode(by_wavelength[0]["y"]), raw[1])


def test_preprocessed_view_shows_x_rows_and_the_transform(
    client: TestClient, fit_run: str
) -> None:
    """The preprocessed view reshapes X.npy rows and names the transform."""
    choose_view(client, files=[SHORT])

    html = client.get("/explore", params={"view": "preprocessed", "segment": "2:1"}).text

    assert f"run {fit_run} の表示ファイル：{SHORT}。" in html
    assert 'data-intensity-transform="sqrt"' in html
    assert "scale = 2" in html
    assert f'data-heatmap-label="{SHORT}"' in html
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
            "file": ["s-00", SHORT],
            "segment": "2:1",
            "wavelength": 400,
            "step_time": 2,
        },
    ).json()
    names = [trace["name"] for trace in trends["by_step_time"]["data"]]
    assert names == ["s-00", SHORT]


def test_run_views_without_a_run_show_a_message(client: TestClient) -> None:
    """Without a succeeded fit run, run-based views explain why nothing is drawn."""
    html = client.get("/explore", params={"view": "preprocessed"}).text

    assert "data-explore-error" in html
    assert "No succeeded fit run" in html
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
    ],
)
def test_invalid_parameters_are_rejected(client: TestClient, params: dict[str, str]) -> None:
    """Unknown views (including the moved component view) and malformed segments
    return 400."""
    assert client.get("/explore", params=params).status_code == 400


def test_trend_rejects_an_unknown_run(client: TestClient) -> None:
    """A trend of an unknown run returns 400."""
    response = client.get(
        "/explore/trend", params={"run": "missing", "wavelength": 0, "step_time": 0}
    )

    assert response.status_code == 400


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

    assert f"run {fit_run} の表示ファイル" in html
    assert 'data-heatmap-label="s-00"' in html


def test_old_succeeded_run_can_be_chosen(
    client: TestClient, settings: Settings, fit_run: str
) -> None:
    """A succeeded run beyond the listed ones is still shown when chosen."""
    choose_view(client, run=fit_run)
    _add_runs(client, settings, 101, "succeeded")

    html = client.get("/explore", params={"view": "preprocessed"}).text
    sidebar = client.get("/sidebar/selection").text

    assert f"run {fit_run} の表示ファイル" in html
    assert 'data-heatmap-label="s-00"' in html
    assert f'<option value="{fit_run}" data-run-kind="fit" selected>' in sidebar


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
    params = {"view": "reconstruction", "segment": "2:1", "k": "2"}
    choose_view(client, files=[SHORT])

    html = client.get("/explore", params=params).text

    assert f'data-heatmap-label="{SHORT} (reconstruction)"' in html
    assert 'name="k" min="1" max="3" value="2"' in html
    assert 'data-intensity-transform="sqrt"' in html
    assert "先頭 1..2 成分の累積再構成" in html
    z = np.array(_decode(_heatmap(html)["z"]), dtype=np.float64)
    assert z.shape == (3, 5)
    assert np.isfinite(z).all()
    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "intensity"

    lines = _trend_values(client, {**params, "file": [SHORT]})
    assert list(lines) == [f"{SHORT} (reconstruction)", f"{SHORT} (preprocessed)"]
    assert np.isfinite(lines[f"{SHORT} (reconstruction)"]).all()
    # X.npy lacks the short file's last Step 2 row.
    assert np.isnan(lines[f"{SHORT} (preprocessed)"]).all()


def test_residual_view_has_no_nan_for_an_imputed_file(
    client: TestClient, fit_run: str
) -> None:
    """Residuals of imputed rows are finite and drawn on a diverging scale."""
    params = {"view": "residual", "segment": "2:1", "k": "1"}
    choose_view(client, files=[SHORT])

    html = client.get("/explore", params=params).text

    assert f'data-heatmap-label="{SHORT}"' in html
    z = np.array(_decode(_heatmap(html)["z"]), dtype=np.float64)
    assert np.isfinite(z).all()
    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "residual"
    assert coloraxis["cmin"] == -coloraxis["cmax"]
    assert np.isfinite(_trend_values(client, {**params, "file": [SHORT]})[SHORT]).all()


def test_residual_plus_reconstruction_is_the_preprocessed_row(
    client: TestClient, fit_run: str
) -> None:
    """Where X.npy has no missing value, residual + reconstruction = X."""
    common = {"file": ["s-00"], "segment": "2:1", "k": "2"}

    reconstruction = _trend_values(client, {**common, "view": "reconstruction"})
    residual = _trend_values(client, {**common, "view": "residual"})["s-00"]

    np.testing.assert_allclose(
        reconstruction["s-00 (reconstruction)"] + residual,
        reconstruction["s-00 (preprocessed)"],
    )


def test_contribution_view_is_the_reconstruction_increment(
    client: TestClient, fit_run: str
) -> None:
    """The k-th contribution is the change of the reconstruction from k-1 to k."""
    common = {"file": ["s-00"], "segment": "2:1"}

    html = client.get(
        "/explore", params={"view": "contribution", "segment": "2:1", "k": "2"}
    ).text
    contribution = _trend_values(client, {**common, "view": "contribution", "k": "2"})
    first = _trend_values(client, {**common, "view": "reconstruction", "k": "1"})
    second = _trend_values(client, {**common, "view": "reconstruction", "k": "2"})

    coloraxis = _embedded(html, "explore-heatmap-figure")["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "contribution"
    assert coloraxis["cmin"] == -coloraxis["cmax"]
    assert "第 2 成分のみの寄与" in html
    np.testing.assert_allclose(
        contribution["s-00"],
        second["s-00 (reconstruction)"] - first["s-00 (reconstruction)"],
    )


def test_drop_strategy_excludes_files_with_missing_values(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Files dropped by ``impute_strategy="drop"`` are listed, not reconstructed."""
    register_fit_run(_workspace(client).database, settings, spectra_paths, "fit-drop", "drop")

    choose_view(client, files=["s-00", SHORT])
    html = client.get("/explore", params={"view": "residual"}).text
    choose_view(client, files=["s-00", SHORT])
    fallback = client.get(
        "/explore", params={"view": "residual", "heatmap_file": SHORT}
    ).text
    choose_view(client, files=[SHORT])
    alone = client.get("/explore", params={"view": "residual"}).text

    assert 'data-heatmap-label="s-00"' in html
    assert "data-dropped" in html
    assert SHORT in html.split("data-dropped", 1)[1].split("</p>", 1)[0]
    assert f'<option value="{SHORT}"' not in html
    assert 'data-heatmap-label="s-00"' in fallback
    assert "heatmap_file=s-00" in fallback
    assert "data-explore-error" in alone
    assert "dropped by the imputation strategy drop" in alone


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
    params = {"view": "contribution", "segment": "2:1", "k": "2"}
    point = {"wavelength": 400, "step_time": 2}
    expected = client.get(
        "/explore/trend",
        params={**params, "run": fit_run, "file": ["s-00", SHORT], **point},
    ).json()
    assert len(expected["by_step_time"]["data"]) == 2
    _clear_cache(client)
    choose_view(client, files=["s-00", SHORT])
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
        "/explore", params={"view": "residual", "segment": "2:1", "k": "1"}
    ).text
    url = _trend_url(html, wavelength=400, step_time=2)
    expected = client.get(url).json()
    assert expected["step_time"] == 2

    _clear_cache(client)

    assert client.get(url).json() == expected


@pytest.fixture
def limited_client(settings: Settings) -> Iterator[TestClient]:
    """Run the application showing up to 2 files."""
    limited = settings.model_copy(update={"ui": UiSettings(explore_max_files=2)})
    with TestClient(create_app(limited)) as opened:
        yield opened


def test_trend_beyond_the_limit_overlays_the_leading_files(
    limited_client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Above ``ui.explore_max_files`` a trend overlays the leading files only."""
    register_fit_run(
        _workspace(limited_client).database, settings, spectra_paths, FIT_RUN_ID, "median"
    )

    response = limited_client.get(
        "/explore/trend",
        params={"file": ["s-10", "s-01", "s-00"], "wavelength": 400, "step_time": 0},
    )

    names = [trace["name"] for trace in response.json()["by_step_time"]["data"]]
    assert names == ["s-00", "s-01"]


def test_trend_at_the_file_limit_prepares_no_row_again(
    limited_client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the limit of files chosen, resolving a trend again reuses the prepared rows."""
    workspace = _workspace(limited_client)
    register_fit_run(workspace.database, settings, spectra_paths, FIT_RUN_ID, "median")
    assert workspace.cache.prepared_row_entries == 2
    choose_view(limited_client, files=["s-00", "s-01"])
    html = limited_client.get("/explore", params={"view": "residual", "k": "1"}).text
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
