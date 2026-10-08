"""Tests for the score screen through FastAPI's ``TestClient``."""

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
from fit_runs import register_shown_run
from spectra import SPECTRA_SHORT_FILE
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import Workspace

Wait = Callable[..., dict[str, object]]
SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _figure(html: str, name: str) -> dict[str, object]:
    """Return the figure embedded in the page's ``<script id=scores-{name}-figure>``."""
    match = re.search(rf'id="scores-{name}-figure">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _decode(values: object) -> list[object]:
    """Decode a Plotly array, which may be a base64 typed array."""
    if isinstance(values, dict):
        return np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=values["dtype"]
        ).tolist()
    return list(values)  # type: ignore[call-overload]


def _traces(html: str, name: str) -> list[dict[str, object]]:
    """Return the traces of one embedded figure."""
    return _figure(html, name)["data"]  # type: ignore[return-value]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application on the fixture workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


def _register(
    client: TestClient,
    settings: Settings,
    spectra_paths: list[Path],
    run_id: str,
    impute_strategy: str,
) -> Path:
    """Register a transform run of a fit run whose files alternate between lots A and B.

    Returns
    -------
    Path
        The run directory.
    """
    metadata = {
        path.stem: {"lot": "AB"[index % 2], "yield_pct": float(index)}
        for index, path in enumerate(spectra_paths)
    }
    register_shown_run(
        _workspace(client).database,
        settings,
        spectra_paths,
        run_id,
        impute_strategy,
        metadata,
    )
    return settings.runs_dir / run_id


@pytest.fixture
def run_dir(client: TestClient, settings: Settings, spectra_paths: list[Path]) -> Path:
    """Register a succeeded transform run of a fit run imputing with the median."""
    return _register(client, settings, spectra_paths, "fit-1", "median")


def _scores_by_stem(run_dir: Path) -> pl.DataFrame:
    """Return ``scores.parquet`` with each file's stem."""
    samples = pl.read_parquet(run_dir / "samples.parquet")
    return pl.read_parquet(run_dir / "scores.parquet").join(
        samples.select("source", "stem"), on="source"
    )


def test_navigation_links_to_the_page(client: TestClient) -> None:
    """The sidebar enables the screen, which reports a missing transform run."""
    html = client.get("/scores").text

    assert 'href="/scores" aria-current="page"' in html
    assert "No succeeded transform run" in html


@pytest.mark.usefixtures("run_dir")
def test_default_page_colors_scores_by_the_default_column(client: TestClient) -> None:
    """Without choices, PC1 against PC2 is colored by ``ui.default_color_by``."""
    response = client.get("/scores")

    assert response.status_code == 200
    html = response.text
    assert 'name="x" form="scores-form" min="1" max="3" value="1"' in html
    assert 'name="y" form="scores-form" min="1" max="3" value="2"' in html
    assert '<option value="lot" selected>' in html
    assert 'name="aggregation"' not in html
    assert 'data-select-url="/sidebar/selection/files/add"' in html
    assert 'data-point-table="point-table"' in html
    assert "data-open-url" not in html
    assert 'name="run"' not in html
    assert 'name="file"' not in html
    traces = _traces(html, "scatter")
    assert [trace["name"] for trace in traces] == ["A", "B"]
    assert _decode(traces[0]["customdata"]) == [f"s-{index:02d}" for index in range(0, 12, 2)]
    layout = _figure(html, "scatter")["layout"]
    assert layout["xaxis"]["title"]["text"] == "PC1"  # type: ignore[index]
    assert "data-explained-variance" not in html
    assert 'id="scores-loadings-figure"' not in html


@pytest.mark.usefixtures("run_dir")
def test_no_color_draws_one_score_trace(client: TestClient) -> None:
    """An empty color choice draws every file in one trace."""
    html = client.get("/scores", params={"color": ""}).text

    assert '<option value="" selected>' in html
    assert len(_traces(html, "scatter")) == 1


def test_scores_match_the_saved_scores(client: TestClient, run_dir: Path) -> None:
    """The score points are the saved scores of the chosen components."""
    html = client.get("/scores", params={"x": 3, "y": 1, "color": ""}).text

    trace = _traces(html, "scatter")[0]
    saved = _scores_by_stem(run_dir)
    by_stem = dict(zip(saved["stem"], saved.select("pca-3", "pca-1").rows(), strict=True))
    points = zip(
        _decode(trace["customdata"]), _decode(trace["x"]), _decode(trace["y"]), strict=True
    )
    for stem, x, y in points:
        np.testing.assert_allclose((x, y), by_stem[stem])


def test_point_table_holds_the_shown_scores(client: TestClient, run_dir: Path) -> None:
    """The table rows hold each file's metadata and the scores of PCm and PCn."""
    html = client.get("/scores", params={"x": 3, "y": 1}).text

    assert re.findall(r"<th>(.*?)</th>", html) == [
        "file", "lot", "date", "yield_pct", "PC3", "PC1"
    ]
    match = re.search(r'data-point-rows>(.*?)</script>', html)
    assert match is not None
    rows = json.loads(match.group(1))
    saved = _scores_by_stem(run_dir)
    by_stem = dict(zip(saved["stem"], saved.select("pca-3", "pca-1").rows(), strict=True))
    assert sorted(row[0] for row in rows) == sorted(by_stem)
    for stem, lot, _, _, x, y in rows:
        assert lot == "AB"[int(stem[2:]) % 2]
        np.testing.assert_allclose((float(x), float(y)), by_stem[stem], rtol=1e-5)
    layout = _figure(html, "scatter")["layout"]
    assert layout["modebar"]["add"] == ["select2d", "lasso2d"]  # type: ignore[index]


def test_trajectories_end_at_the_saved_scores(client: TestClient, run_dir: Path) -> None:
    """Each trajectory ends at the file's score, including the imputed file."""
    stems = ["s-00", SHORT]
    choose_view(client, files=stems)
    html = client.get("/scores", params={"x": 1, "y": 2}).text

    traces = _traces(html, "trajectories")
    assert [trace["name"] for trace in traces] == stems
    saved = _scores_by_stem(run_dir)
    for trace in traces:
        expected = saved.filter(pl.col("stem") == trace["name"]).select("pca-1", "pca-2")
        np.testing.assert_allclose(
            (_decode(trace["x"])[-1], _decode(trace["y"])[-1]), expected.row(0), rtol=1e-5
        )
    # Step 1 has four time points and Step 2 three.
    assert trace["text"][0] == "(1, 1, 0)"
    assert len(trace["text"]) == 7
    assert "data-dropped" not in html


def test_drop_run_reports_files_without_trajectory(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Under ``impute_strategy="drop"``, the file with a missing value is reported."""
    _register(client, settings, spectra_paths, "fit-drop", "drop")

    choose_view(client, files=["s-00", SHORT])
    html = client.get("/scores").text

    assert f"Cannot draw trajectories because they contain missing values and are dropped by the imputation strategy drop: {SHORT}" in html
    assert [trace["name"] for trace in _traces(html, "trajectories")] == ["s-00"]
    scored = [
        stem for trace in _traces(html, "scatter") for stem in _decode(trace["customdata"])
    ]
    assert SHORT not in scored


@pytest.mark.usefixtures("run_dir")
def test_out_of_range_components_fall_back_to_the_defaults(client: TestClient) -> None:
    """Unavailable component numbers are replaced by PC1 and PC2."""
    html = client.get("/scores", params={"x": 9, "y": 0}).text

    assert 'name="x" form="scores-form" min="1" max="3" value="1"' in html
    assert 'name="y" form="scores-form" min="1" max="3" value="2"' in html


@pytest.mark.usefixtures("run_dir")
@pytest.mark.parametrize("params", [{"x": "one"}])
def test_invalid_parameters_are_rejected(
    client: TestClient, params: dict[str, str]
) -> None:
    """Non-integer components are client errors."""
    response = client.get("/scores", params=params)

    assert response.status_code in (400, 422)
