"""Tests for the model screen through FastAPI's ``TestClient``."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from fit_runs import register_fit_run

from flat_pca.feature_engineering.pca import PcaModel
from flat_pca.webui.app import create_app
from flat_pca.webui.services.display_cache import DisplayCache
from flat_pca.webui.services.fit_artifacts import load_pca_model
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import Workspace


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


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application on the fixture workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


@pytest.fixture
def run_dir(client: TestClient, settings: Settings, spectra_paths: list[Path]) -> Path:
    """Register a succeeded fit run imputing missing values with the median."""
    register_fit_run(_workspace(client).database, settings, spectra_paths, "fit-1", "median")
    return settings.runs_dir / "fit-1"


def test_navigation_links_to_the_page(client: TestClient) -> None:
    """The sidebar enables the screen, which reports a missing fit run."""
    html = client.get("/model").text

    assert 'href="/model" aria-current="page"' in html
    assert "data-model-error" in html
    assert "No succeeded fit run" in html
    assert "data-model-settings" not in html
    response = client.get("/model/trend", params={"wavelength": 0, "step_time": 0})
    assert response.status_code == 404


@pytest.mark.usefixtures("run_dir")
def test_default_page_draws_every_figure_from_the_run(client: TestClient) -> None:
    """Without choices, the run's explained variance, loadings, and PC1 are drawn."""
    response = client.get("/model")

    assert response.status_code == 200
    html = response.text
    assert 'name="run"' not in html
    assert 'data-model-run="fit-1"' in html
    assert 'name="x" form="model-form" min="1" max="3" value="1"' in html
    assert 'name="y" form="model-form" min="1" max="3" value="2"' in html
    assert '<option value="rms" selected>' in html
    assert 'name="k" form="model-form" min="1" max="3" value="1"' in html
    assert '<option value="1:1" selected>' in html
    assert 'name="file"' not in html
    assert html.count("<td>PC") == 3
    for name in ("scree", "loadings"):
        assert f'id="model-{name}-figure"' in html
    assert 'data-heatmap-label="PC1"' in html
    assert (
        'data-trend-url="/model/trend?run=fit-1&amp;view=component&amp;k=1&amp;segment=1%3A1"'
        in html
    )
    assert '<option value="component" selected>PCA component k</option>' in html
    for key in ("mean", "scaling_center", "scaling_scale", "impute_median"):
        assert f'<option value="{key}" >' in html
    assert 'value="outlier_lower"' not in html
    assert "data-model-notice" not in html


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ("mean", lambda model: model.pca.mean_),
        (
            "scaling_center",
            lambda model: [model.scaling_model.centers[c] for c in model.columns],
        ),
        (
            "scaling_scale",
            lambda model: [model.scaling_model.scales[c] for c in model.columns],
        ),
        (
            "impute_median",
            lambda model: [model.impute_model.values[c] for c in model.columns],
        ),
    ],
)
def test_parameter_heatmap_reshapes_the_model_values(
    client: TestClient,
    run_dir: Path,
    view: str,
    expected: Callable[[PcaModel], object],
) -> None:
    """A preprocessing parameter's heatmap and trends reshape the model's values."""
    html = client.get("/model", params={"view": view, "segment": "2:1"}).text

    assert f'<option value="{view}" selected>' in html
    assert (
        f'data-trend-url="/model/trend?run=fit-1&amp;view={view}&amp;k=1&amp;segment=2%3A1"'
        in html
    )
    figure = _embedded(html, "explore-heatmap-figure")
    assert figure["layout"]["coloraxis"]["colorbar"]["title"]["text"] == "value"  # type: ignore[index]
    model = load_pca_model(run_dir)
    values = np.asarray(expected(model), dtype=np.float64)
    features = pl.read_parquet(run_dir / "features.parquet").with_columns(
        pl.Series("value", values)
    )
    grid = (
        features.filter((pl.col("Step") == 2) & (pl.col("Sequence") == 1))
        .pivot(on="wavelength", index="StepTime", values="value", sort_columns=True)
        .sort("StepTime")
    )
    np.testing.assert_allclose(
        np.array(_decode(figure["data"][0]["z"])),  # type: ignore[index]
        grid.drop("StepTime").to_numpy(),
    )

    trends = client.get(
        "/model/trend",
        params={"run": "fit-1", "view": view, "k": 1, "segment": "2:1"}
        | {"wavelength": 400, "step_time": 0},
    ).json()
    label = trends["by_step_time"]["data"][0]["name"]
    assert f'data-heatmap-label="{label}"' in html
    np.testing.assert_allclose(
        _decode(trends["by_wavelength"]["data"][0]["y"]),
        grid.drop("StepTime").to_numpy()[0],
    )


@pytest.mark.usefixtures("run_dir")
def test_page_shows_the_model_chosen_on_the_transform_screen(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The newest fit run is shown until another model is chosen on the transform screen."""
    register_fit_run(_workspace(client).database, settings, spectra_paths, "fit-2", "median")
    assert 'data-model-run="fit-2"' in client.get("/model").text

    response = client.post("/transform/settings", data={"model": "fit-1", "use_same_data": "true"})

    assert response.status_code == 200
    html = client.get("/model").text
    assert 'data-model-run="fit-1"' in html
    assert 'data-model-settings="fit-1"' in html
    assert html.index('data-model-run="fit-1"') < html.index("data-model-settings")
    assert "<span data-model-settings-headline>3 components · scaling z-score</span>" in html
    assert 'href="/fit?run=fit-1" data-model-settings-link' in html


@pytest.mark.parametrize("view", ["scaling_scale", "impute_median", "outlier_lower"])
def test_parameters_absent_from_the_run_fall_back_to_the_component(
    client: TestClient, settings: Settings, spectra_paths: list[Path], view: str
) -> None:
    """A run without scaling, imputation values, or outlier handling does not offer them."""
    register_fit_run(
        _workspace(client).database,
        settings,
        spectra_paths,
        "fit-plain",
        "drop",
        scaling_strategy="none",
    )

    html = client.get("/model", params={"view": view}).text

    assert f'value="{view}"' not in html
    assert '<option value="component" selected>' in html
    assert '<option value="mean" >' in html
    assert "data-model-notice" in html
    assert "This run has no" in html
    assert 'data-heatmap-label="PC1"' in html


@pytest.mark.parametrize(
    ("aggregation", "reduce"),
    [
        ("mean", np.mean),
        ("rms", lambda values: np.sqrt(np.mean(values**2))),
        ("abs_mean", lambda values: np.mean(np.abs(values))),
    ],
)
def test_loadings_aggregate_the_components_by_wavelength(
    client: TestClient,
    run_dir: Path,
    aggregation: str,
    reduce: Callable[[np.ndarray], float],
) -> None:
    """Each loading point aggregates one wavelength's coefficients."""
    html = client.get("/model", params={"x": 2, "y": 3, "aggregation": aggregation}).text

    trace = _embedded(html, "model-loadings-figure")["data"][0]  # type: ignore[index]
    wavelengths = pl.read_parquet(run_dir / "features.parquet")["wavelength"].to_numpy()
    components = np.load(run_dir / "components.npy").astype(np.float64)
    expected = sorted(set(wavelengths.tolist()))
    assert _decode(trace["customdata"]) == expected
    for index, wavelength in enumerate(expected):
        mask = wavelengths == wavelength
        assert _decode(trace["x"])[index] == pytest.approx(reduce(components[1, mask]))
        assert _decode(trace["y"])[index] == pytest.approx(reduce(components[2, mask]))


def test_component_heatmap_reshapes_the_chosen_component(
    client: TestClient, run_dir: Path
) -> None:
    """The heatmap shows component k on a diverging scale, and its trends match it."""
    html = client.get("/model", params={"k": "2", "segment": "2:1"}).text

    assert 'name="k" form="model-form" min="1" max="3" value="2"' in html
    assert '<option value="2:1" selected>' in html
    assert 'data-heatmap-label="PC2"' in html
    figure = _embedded(html, "explore-heatmap-figure")
    coloraxis = figure["layout"]["coloraxis"]  # type: ignore[index]
    assert coloraxis["colorbar"]["title"]["text"] == "coefficient"
    assert coloraxis["cmin"] == -coloraxis["cmax"]

    trends = client.get(
        "/model/trend",
        params={"run": "fit-1", "k": 2, "segment": "2:1", "wavelength": 400, "step_time": 0},
    ).json()
    assert [trace["name"] for trace in trends["by_step_time"]["data"]] == ["PC2"]
    features = pl.read_parquet(run_dir / "features.parquet").with_row_index("position")
    at_time = features.filter(
        (pl.col("Step") == 2) & (pl.col("Sequence") == 1) & (pl.col("StepTime") == 0)
    ).sort("wavelength")
    components = np.load(run_dir / "components.npy").astype(np.float64)
    by_wavelength = trends["by_wavelength"]["data"][0]
    np.testing.assert_allclose(
        _decode(by_wavelength["y"]), components[1, at_time["position"].to_numpy()]
    )


@pytest.mark.usefixtures("run_dir")
def test_trend_without_a_kept_matrix_resolves_the_screen(client: TestClient) -> None:
    """A trend request no page drew falls back to the defaults and is resolved."""
    workspace = _workspace(client)
    workspace.cache = DisplayCache(workspace.settings.ui.explore_max_files)

    trends = client.get("/model/trend", params={"wavelength": 400, "step_time": 0}).json()

    assert [trace["name"] for trace in trends["by_step_time"]["data"]] == ["PC1"]


@pytest.mark.usefixtures("run_dir")
def test_out_of_range_choices_fall_back_to_the_defaults(client: TestClient) -> None:
    """Unavailable component numbers and segments are replaced by the defaults."""
    html = client.get("/model", params={"x": 9, "y": 0, "k": 4, "segment": "9:9"}).text

    assert 'name="x" form="model-form" min="1" max="3" value="1"' in html
    assert 'name="y" form="model-form" min="1" max="3" value="2"' in html
    assert 'name="k" form="model-form" min="1" max="3" value="1"' in html
    assert '<option value="1:1" selected>' in html


@pytest.mark.usefixtures("run_dir")
@pytest.mark.parametrize(
    "params",
    [
        {"aggregation": "median"},
        {"segment": "a:b"},
        {"k": "one"},
        {"view": "scale"},
    ],
)
def test_invalid_parameters_are_rejected(client: TestClient, params: dict[str, str]) -> None:
    """Unknown aggregations and views, malformed segments, and non-integers are rejected."""
    assert client.get("/model", params=params).status_code in (400, 422)
