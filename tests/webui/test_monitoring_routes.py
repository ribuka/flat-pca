"""Tests for the T² and Q screen and the Q contribution view."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from fit_runs import register_shown_run
from spectra import SPECTRA_FILE_COUNT, SPECTRA_SHORT_FILE
from view_choice import choose_view

from flat_pca.webui.app import create_app
from flat_pca.webui.services.explore import ExploreRequest, resolve_explore
from flat_pca.webui.services.runs import get_run, list_succeeded_runs
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import TRANSFORM_JOB, Workspace

SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"
# Files are dated in reverse index order; this one has no date.
UNDATED = "s-05"


def _workspace(client: TestClient) -> Workspace:
    """Return the application's workspace."""
    return client.app.state.workspace


def _figure(html: str, element_id: str) -> dict[str, object]:
    """Return the figure embedded in the page's ``<script id={element_id}>``."""
    match = re.search(rf'id="{element_id}">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _decode(values: object) -> list[object]:
    """Decode a Plotly array, which may be a base64 typed array."""
    if isinstance(values, dict):
        return np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=values["dtype"]
        ).tolist()
    return list(values)  # type: ignore[call-overload]


def _points(html: str, name: str) -> dict[str, tuple[object, float]]:
    """Return ``(x, y)`` of every point of a T²/Q figure keyed by stem.

    ``x`` is a number, or the ISO text of a date on a date axis.
    """
    points: dict[str, tuple[object, float]] = {}
    for trace in _figure(html, f"monitoring-{name}-figure")["data"]:  # type: ignore[union-attr]
        for stem, x, y in zip(
            _decode(trace["customdata"]), _decode(trace["x"]), _decode(trace["y"]), strict=True
        ):
            points[str(stem)] = (x, float(y))
    return points


def _x_order(html: str, name: str) -> list[str]:
    """Return the stems of a control chart sorted by their horizontal position."""
    positions = _points(html, name)
    return sorted(positions, key=lambda stem: positions[stem][0])  # type: ignore[arg-type, return-value]


def _as_category(html: str) -> str:
    """Return the ``As category`` checkbox of the page."""
    match = re.search(r'<input type="checkbox" name="as_category"[^>]*>', html)
    assert match is not None
    return match.group(0)


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
    """Register a transform run of a fit run whose files are dated in reverse index order.

    Returns
    -------
    Path
        The run directory.
    """
    metadata = {
        path.stem: {
            "lot": "AB"[index % 2],
            "yield_pct": 80.0 + index,
            "date": None
            if path.stem == UNDATED
            else f"2024-01-{SPECTRA_FILE_COUNT - index:02d}T00:00:00",
        }
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
    html = client.get("/monitoring").text

    assert 'href="/monitoring" aria-current="page"' in html
    assert "No succeeded transform run" in html


@pytest.mark.usefixtures("run_dir")
def test_default_x_axis_draws_the_default_date_column(client: TestClient) -> None:
    """Without a choice, the charts draw ``ui.default_x_axis`` on a date axis."""
    response = client.get("/monitoring")

    assert response.status_code == 200
    html = response.text
    assert '<option value="date" selected>' in html
    assert "checked" not in _as_category(html)
    assert "disabled" not in _as_category(html)
    assert 'data-plot="monitoring-q-figure" data-point-table="point-table"' in html
    assert "data-select-url" not in html
    assert "data-open-url" not in html
    assert "run fit-1 の T² と Q です。" in html
    assert "管理図の横軸は date の値です。" in html
    assert 'name="run"' not in html
    # The last file has the earliest date; the undated file has no position.
    expected = [f"s-{index:02d}" for index in reversed(range(SPECTRA_FILE_COUNT))]
    expected.remove(UNDATED)
    for name in ("t2", "q"):
        assert _x_order(html, name) == expected
    positions = _points(html, "q")
    assert positions["s-00"][0] == f"2024-01-{SPECTRA_FILE_COUNT:02d}T00:00:00"
    layout = _figure(html, "monitoring-q-figure")["layout"]
    assert layout["xaxis"]["title"]["text"] == "date"  # type: ignore[index]
    unplotted = html.split("data-unplotted>", 1)[1].split("</p>", 1)[0]
    assert unplotted == f"Not in the control charts because date is missing: {UNDATED}"
    # The scatter plot and the UCL summary still hold the undated file.
    assert UNDATED in _points(html, "scatter")
    assert f"/ {SPECTRA_FILE_COUNT} files" in html


@pytest.mark.usefixtures("run_dir")
def test_numeric_column_is_drawn_at_its_values(client: TestClient) -> None:
    """A numeric column places each point at its value."""
    html = client.get("/monitoring", params={"x_axis": "yield_pct"}).text

    positions = _points(html, "t2")
    assert {stem: x for stem, (x, _) in positions.items()} == {
        f"s-{index:02d}": 80.0 + index for index in range(SPECTRA_FILE_COUNT)
    }
    assert "data-unplotted" not in html


@pytest.mark.usefixtures("run_dir")
def test_numeric_column_as_category_draws_the_rank(client: TestClient) -> None:
    """``As category`` draws the files at their rank, with missing values last."""
    html = client.get("/monitoring", params={"x_axis": "date", "as_category": "true"}).text

    assert "checked" in _as_category(html)
    assert "管理図は date の昇順" in html
    expected = [f"s-{index:02d}" for index in reversed(range(SPECTRA_FILE_COUNT))]
    expected.remove(UNDATED)
    positions = _points(html, "q")
    assert _x_order(html, "q") == [*expected, UNDATED]
    assert sorted(x for x, _ in positions.values()) == list(range(1, SPECTRA_FILE_COUNT + 1))
    layout = _figure(html, "monitoring-q-figure")["layout"]
    assert layout["xaxis"]["title"]["text"] == "file order (date)"  # type: ignore[index]
    assert "data-unplotted" not in html


@pytest.mark.usefixtures("run_dir")
@pytest.mark.parametrize("as_category", ["", "true"])
def test_category_column_disables_as_category(client: TestClient, as_category: str) -> None:
    """A category column is always drawn by rank and disables ``As category``."""
    params = {"x_axis": "lot"} | ({"as_category": as_category} if as_category else {})
    html = client.get("/monitoring", params=params).text

    assert "disabled" in _as_category(html)
    positions = _points(html, "t2")
    assert sorted(x for x, _ in positions.values()) == list(range(1, SPECTRA_FILE_COUNT + 1))


@pytest.mark.usefixtures("run_dir")
def test_natural_order_on_request(client: TestClient) -> None:
    """An empty x axis choice sorts the files by their names."""
    html = client.get("/monitoring", params={"x_axis": ""}).text

    assert '<option value="" selected>' in html
    assert "disabled" in _as_category(html)
    assert _x_order(html, "t2") == [f"s-{index:02d}" for index in range(SPECTRA_FILE_COUNT)]


def test_figures_show_the_saved_statistics(client: TestClient, run_dir: Path) -> None:
    """The charts and the scatter plot show the saved T², Q, and limits."""
    html = client.get("/monitoring", params={"color": ""}).text

    saved = _scores_by_stem(run_dir)
    t2 = dict(zip(saved["stem"], saved["mahalanobis_sq"], strict=True))
    q = dict(zip(saved["stem"], saved["spe"], strict=True))
    for stem, (_, value) in _points(html, "t2").items():
        assert value == pytest.approx(t2[stem])
    for stem, (_, value) in _points(html, "q").items():
        assert value == pytest.approx(q[stem])
    for stem, (x, y) in _points(html, "scatter").items():
        assert (float(x), y) == pytest.approx((t2[stem], q[stem]))  # type: ignore[arg-type]
    shapes = _figure(html, "monitoring-scatter-figure")["layout"]["shapes"]  # type: ignore[index]
    assert shapes[0]["x0"] == pytest.approx(saved["mahalanobis_ucl"][0])
    assert shapes[1]["y0"] == pytest.approx(saved["spe_ucl"][0])
    exceeding = saved.filter(pl.col("spe_exceeds_ucl"))["stem"].to_list()
    assert f"Q {len(exceeding)}" in html.split("data-exceeding", 1)[1].split("</p>", 1)[0]
    # Points above a UCL are drawn like the others, in one trace per figure.
    for name in ("t2", "q", "scatter"):
        assert len(_figure(html, f"monitoring-{name}-figure")["data"]) == 1  # type: ignore[arg-type]


@pytest.mark.usefixtures("run_dir")
def test_points_are_colored_by_the_default_column(client: TestClient) -> None:
    """Without a choice, all three figures are colored by ``ui.default_color_by``."""
    html = client.get("/monitoring").text

    assert '<select name="color">' in html
    assert '<option value="lot" selected>' in html
    for name in ("t2", "q", "scatter"):
        figure = _figure(html, f"monitoring-{name}-figure")
        names = {trace["name"] for trace in figure["data"]}  # type: ignore[index, union-attr]
        assert names == {"A", "B"}
        assert figure["layout"]["legend"]["title"]["text"] == "lot"  # type: ignore[index]

    uncolored = client.get("/monitoring", params={"color": ""}).text
    assert '<option value="" selected>none</option>' in uncolored
    (trace,) = _figure(uncolored, "monitoring-q-figure")["data"]  # type: ignore[misc]
    assert "name" not in trace


def _point_rows(html: str) -> list[list[str]]:
    """Return the rows embedded for the page's point table."""
    match = re.search(r"<script type=\"application/json\" data-point-rows>(.*?)</script>", html)
    assert match is not None
    return json.loads(match.group(1))


def test_point_table_holds_the_statistics_of_every_file(
    client: TestClient, run_dir: Path
) -> None:
    """The table rows hold each file's metadata, T², Q, and UCL excess, in chart order."""
    html = client.get("/monitoring").text

    headers = re.findall(r"<th>(.*?)</th>", html)
    assert headers == [
        "file", "lot", "date", "yield_pct", "T²", "Q", "T² above UCL", "Q above UCL"
    ]
    for name in ("t2", "scatter"):
        assert f'data-plot="monitoring-{name}-figure" data-point-table="point-table"' in html
    rows = _point_rows(html)
    # In chart order, followed by the undated file missing from the charts.
    assert [row[0] for row in rows] == [*_x_order(html, "q"), UNDATED]
    saved = _scores_by_stem(run_dir)
    for row in rows:
        record = saved.row(by_predicate=pl.col("stem") == row[0], named=True)
        assert float(row[4]) == pytest.approx(record["mahalanobis_sq"], rel=1e-5)
        assert float(row[5]) == pytest.approx(record["spe"], rel=1e-5)
        assert row[7] == ("yes" if record["spe_exceeds_ucl"] else "")
    undated = next(row for row in rows if row[0] == UNDATED)
    assert undated[2] == ""
    layout = _figure(html, "monitoring-t2-figure")["layout"]
    assert layout["modebar"]["add"] == ["select2d", "lasso2d"]  # type: ignore[index]


def test_drop_run_reports_files_without_statistics(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Under ``impute_strategy="drop"``, the file with a missing value is reported."""
    _register(client, settings, spectra_paths, "fit-drop", "drop")

    html = client.get("/monitoring").text

    assert f"dropped by the imputation strategy drop: {SHORT}" in html
    assert SHORT not in _points(html, "scatter")


@pytest.mark.usefixtures("run_dir")
def test_page_shows_the_run_chosen_on_the_transform_screen(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """The page shows the chosen transform run, not the newest one, once it is chosen."""
    _register(client, settings, spectra_paths, "fit-2", "median")
    assert "run fit-2 の T² と Q です。" in client.get("/monitoring").text

    choose_view(client, run="fit-1")

    assert "run fit-1 の T² と Q です。" in client.get("/monitoring").text


@pytest.mark.usefixtures("run_dir")
def test_unknown_x_axis_column_falls_back_to_the_default(client: TestClient) -> None:
    """An unavailable horizontal-axis column is replaced by the default one."""
    html = client.get("/monitoring", params={"x_axis": "missing"}).text

    assert '<option value="date" selected>' in html


@pytest.mark.usefixtures("run_dir")
def test_q_contribution_page_shows_the_fixed_component_count(client: TestClient) -> None:
    """The view reconstructs from the run's Q component count without a k input."""
    choose_view(client, files=[SHORT])

    response = client.get("/explore", params={"view": "q_contribution", "k": 3})

    assert response.status_code == 200
    html = response.text
    assert '<option value="q_contribution" selected>' in html
    assert 'name="k"' not in html
    assert "先頭 1..2 成分" in html
    assert f'data-heatmap-label="{SHORT}"' in html
    layout = _figure(html, "explore-heatmap-figure")["layout"]
    assert layout["coloraxis"]["colorbar"]["title"]["text"] == "q_contribution"  # type: ignore[index]
    # The float32 artifacts reproduce the saved Q of these spectra.
    assert "data-q-mismatch" not in html


@pytest.mark.usefixtures("run_dir")
@pytest.mark.parametrize(
    ("path", "params"),
    [("/monitoring", {}), ("/explore", {"view": "q_contribution"})],
)
def test_invalid_saved_statistics_ask_to_run_again(
    client: TestClient, path: str, params: dict[str, str]
) -> None:
    """Broken T² and Q settings of the run are shown as an error, not a server error."""
    choose_view(client, files=[SHORT])
    database = _workspace(client).database
    run = get_run(database, "fit-1")
    assert run is not None
    config = json.loads(str(run["config_json"]))
    config["spe"] = None
    database.execute(
        "UPDATE runs SET config_json = ? WHERE run_id = ?", [json.dumps(config), "fit-1"]
    )

    response = client.get(path, params=params)

    assert response.status_code == 200
    assert "run the fit again" in response.text


def _write_offset_spectra(directory: Path) -> list[Path]:
    """Write spectra with small variations on a large baseline.

    Rounding them to float32 loses the variations between the files.

    Returns
    -------
    list[Path]
        ``s-{index:02d}.parquet`` paths in index order.
    """
    directory.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    paths = []
    for index in range(20):
        values = 1e8 + rng.normal(size=(20, 4))
        frame = pl.DataFrame(
            {
                "Time": np.arange(20, dtype=np.float64),
                "Step": [1] * 10 + [2] * 10,
                "Sequence": [1] * 20,
                **{f"{400.0 + column:.1f}nm": values[:, column] for column in range(4)},
            }
        )
        path = directory / f"s-{index:02d}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths


def test_q_contribution_page_warns_when_float32_artifacts_lose_the_saved_q(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """Contributions from float32 artifacts that miss the saved Q are reported."""
    register_shown_run(
        _workspace(client).database,
        settings,
        _write_offset_spectra(tmp_path / "offset"),
        "fit-1",
        "median",
    )
    run_dir = settings.runs_dir / "fit-1"
    choose_view(client, files=["s-01"])

    html = client.get("/explore", params={"view": "q_contribution"}).text

    saved = _scores_by_stem(run_dir).filter(pl.col("stem") == "s-01")["spe"][0]
    match = re.search(r'data-q-mismatch="s-01">(.*?)</p>', html, re.DOTALL)
    assert match is not None
    assert f"{saved:.8g}" in match.group(1)
    assert 'Run the fit again with jobs.artifact_dtype = "float64".' in match.group(1)


@pytest.mark.parametrize("stem", ["s-00", SHORT])
def test_q_contributions_add_up_to_the_saved_q(
    client: TestClient, run_dir: Path, stem: str
) -> None:
    """The unbinned contributions of every segment add up to the file's Q.

    ``SHORT`` lacks its last time point, which the median imputation fills.
    """
    workspace = _workspace(client)
    transform_runs = list_succeeded_runs(workspace.database, TRANSFORM_JOB, None)
    first = resolve_explore(
        workspace.cache,
        transform_runs,
        ExploreRequest(view="q_contribution", files=(stem,)),
        workspace.settings.ui.explore_max_files,
    )
    total = 0.0
    for step, sequence in first.segment_options:
        shown = resolve_explore(
            workspace.cache,
            transform_runs,
            ExploreRequest(
                view="q_contribution", files=(stem,), segment=f"{step}:{sequence}"
            ),
            workspace.settings.ui.explore_max_files,
        )
        assert shown.error is None
        total += float(np.nansum(shown.matrices[stem].values))

    saved = _scores_by_stem(run_dir).filter(pl.col("stem") == stem)["spe"][0]
    assert total == pytest.approx(saved, rel=1e-5)


@pytest.mark.usefixtures("run_dir")
def test_points_can_be_colored_by_file_name(client: TestClient) -> None:
    """Color by offers the file name, which the x axis choices do not."""
    html = client.get("/monitoring", params={"color": "stem"}).text

    assert '<option value="stem" selected>file name</option>' in html
    x_axis = html.split('name="x_axis"', 1)[1].split("</select>", 1)[0]
    assert 'value="stem"' not in x_axis
    scatter = _figure(html, "monitoring-scatter-figure")
    names = {trace["name"] for trace in scatter["data"]}  # type: ignore[index, union-attr]
    assert names == {f"s-{index:02d}" for index in range(SPECTRA_FILE_COUNT)}
    assert scatter["layout"]["legend"]["title"]["text"] == "stem"  # type: ignore[index]
