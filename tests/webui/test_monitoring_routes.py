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
from fit_runs import register_fit_run
from spectra import SPECTRA_FILE_COUNT, SPECTRA_SHORT_FILE

from flat_pca.webui.app import create_app
from flat_pca.webui.services.explore import ExploreRequest, resolve_explore
from flat_pca.webui.services.runs import list_succeeded_runs
from flat_pca.webui.settings import Settings
from flat_pca.webui.workspace import FIT_JOB, Workspace

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


def _points(html: str, name: str) -> dict[str, tuple[float, float]]:
    """Return ``(x, y)`` of every point of a T²/Q figure keyed by stem."""
    points: dict[str, tuple[float, float]] = {}
    for trace in _figure(html, f"monitoring-{name}-figure")["data"]:  # type: ignore[union-attr]
        for stem, x, y in zip(
            _decode(trace["customdata"]), _decode(trace["x"]), _decode(trace["y"]), strict=True
        ):
            points[str(stem)] = (float(x), float(y))
    return points


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
    """Register a fit run whose files are dated in reverse index order.

    Returns
    -------
    Path
        The run directory.
    """
    metadata = {
        path.stem: {
            "lot": "AB"[index % 2],
            "date": None
            if path.stem == UNDATED
            else f"2024-01-{SPECTRA_FILE_COUNT - index:02d}T00:00:00",
        }
        for index, path in enumerate(spectra_paths)
    }
    register_fit_run(
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
    """Register a succeeded fit run imputing missing values with the median."""
    return _register(client, settings, spectra_paths, "fit-1", "median")


def _scores_by_stem(run_dir: Path) -> pl.DataFrame:
    """Return ``scores.parquet`` with each file's stem."""
    samples = pl.read_parquet(run_dir / "samples.parquet")
    return pl.read_parquet(run_dir / "scores.parquet").join(
        samples.select("source", "stem"), on="source"
    )


def test_navigation_links_to_the_page(client: TestClient) -> None:
    """The sidebar enables the screen, which reports a missing fit run."""
    html = client.get("/monitoring").text

    assert 'href="/monitoring" aria-current="page"' in html
    assert "成功した fit run がありません" in html


@pytest.mark.usefixtures("run_dir")
def test_default_order_follows_the_default_column(client: TestClient) -> None:
    """Without a choice, the charts follow ``ui.default_order_by``."""
    response = client.get("/monitoring")

    assert response.status_code == 200
    html = response.text
    assert '<option value="date" selected>' in html
    assert 'data-explore-url="/explore?view=q_contribution&amp;run=fit-1"' in html
    # The last file has the earliest date and the undated file comes last.
    expected = [f"s-{index:02d}" for index in reversed(range(SPECTRA_FILE_COUNT))]
    expected.remove(UNDATED)
    positions = _points(html, "q")
    assert sorted(positions, key=lambda stem: positions[stem][0]) == [*expected, UNDATED]
    assert sorted(positions[stem][0] for stem in positions) == list(
        range(1, SPECTRA_FILE_COUNT + 1)
    )


@pytest.mark.usefixtures("run_dir")
def test_natural_order_on_request(client: TestClient) -> None:
    """An empty order choice sorts the files by their names."""
    html = client.get("/monitoring", params={"order": ""}).text

    assert '<option value="" selected>' in html
    positions = _points(html, "t2")
    assert sorted(positions, key=lambda stem: positions[stem][0]) == [
        f"s-{index:02d}" for index in range(SPECTRA_FILE_COUNT)
    ]


def test_figures_show_the_saved_statistics(client: TestClient, run_dir: Path) -> None:
    """The charts and the scatter plot show the saved T², Q, and limits."""
    html = client.get("/monitoring").text

    saved = _scores_by_stem(run_dir)
    t2 = dict(zip(saved["stem"], saved["mahalanobis_sq"], strict=True))
    q = dict(zip(saved["stem"], saved["spe"], strict=True))
    for stem, (_, value) in _points(html, "t2").items():
        assert value == pytest.approx(t2[stem])
    for stem, (_, value) in _points(html, "q").items():
        assert value == pytest.approx(q[stem])
    for stem, (x, y) in _points(html, "scatter").items():
        assert (x, y) == pytest.approx((t2[stem], q[stem]))
    shapes = _figure(html, "monitoring-scatter-figure")["layout"]["shapes"]  # type: ignore[index]
    assert shapes[0]["x0"] == pytest.approx(saved["mahalanobis_ucl"][0])
    assert shapes[1]["y0"] == pytest.approx(saved["spe_ucl"][0])
    exceeding = saved.filter(pl.col("spe_exceeds_ucl"))["stem"].to_list()
    assert f"Q {len(exceeding)} 件" in html
    q_traces = _figure(html, "monitoring-q-figure")["data"]  # type: ignore[index]
    assert sorted(_decode(q_traces[1]["customdata"])) == sorted(exceeding)  # type: ignore[index]


def test_drop_run_reports_files_without_statistics(
    client: TestClient, settings: Settings, spectra_paths: list[Path]
) -> None:
    """Under ``impute_strategy="drop"``, the file with a missing value is reported."""
    _register(client, settings, spectra_paths, "fit-drop", "drop")

    html = client.get("/monitoring").text

    assert f"T²・Q を持たないファイル：{SHORT}" in html
    assert SHORT not in _points(html, "scatter")


@pytest.mark.usefixtures("run_dir")
@pytest.mark.parametrize("params", [{"run": "missing"}])
def test_invalid_parameters_are_rejected(
    client: TestClient, params: dict[str, str]
) -> None:
    """Unknown runs are client errors."""
    assert client.get("/monitoring", params=params).status_code == 400


@pytest.mark.usefixtures("run_dir")
def test_unknown_order_column_falls_back_to_the_default(client: TestClient) -> None:
    """An unavailable ordering column is replaced by the default one."""
    html = client.get("/monitoring", params={"order": "missing"}).text

    assert '<option value="date" selected>' in html


@pytest.mark.usefixtures("run_dir")
def test_q_contribution_page_shows_the_fixed_component_count(client: TestClient) -> None:
    """The view reconstructs from the run's Q component count without a k input."""
    response = client.get(
        "/explore", params={"view": "q_contribution", "run": "fit-1", "file": SHORT, "k": 3}
    )

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
    register_fit_run(
        _workspace(client).database,
        settings,
        _write_offset_spectra(tmp_path / "offset"),
        "fit-1",
        "median",
    )
    run_dir = settings.runs_dir / "fit-1"

    html = client.get(
        "/explore", params={"view": "q_contribution", "run": "fit-1", "file": "s-01"}
    ).text

    saved = _scores_by_stem(run_dir).filter(pl.col("stem") == "s-01")["spe"][0]
    match = re.search(r'data-q-mismatch="s-01">(.*?)</p>', html, re.DOTALL)
    assert match is not None
    assert f"{saved:.8g}" in match.group(1)
    assert 'jobs.artifact_dtype = "float64" で再実行してください' in match.group(1)


@pytest.mark.parametrize("stem", ["s-00", SHORT])
def test_q_contributions_add_up_to_the_saved_q(
    client: TestClient, run_dir: Path, stem: str
) -> None:
    """The unbinned contributions of every segment add up to the file's Q.

    ``SHORT`` lacks its last time point, which the median imputation fills.
    """
    workspace = _workspace(client)
    fit_runs = list_succeeded_runs(workspace.database, FIT_JOB, None)
    first = resolve_explore(
        workspace.database,
        [],
        workspace.cache,
        fit_runs,
        ExploreRequest(view="q_contribution", files=(stem,)),
    )
    total = 0.0
    for step, sequence in first.segment_options:
        shown = resolve_explore(
            workspace.database,
            [],
            workspace.cache,
            fit_runs,
            ExploreRequest(
                view="q_contribution", files=(stem,), segment=f"{step}:{sequence}"
            ),
        )
        assert shown.error is None
        total += float(np.nansum(shown.matrices[stem].values))

    saved = _scores_by_stem(run_dir).filter(pl.col("stem") == stem)["spe"][0]
    assert total == pytest.approx(saved, rel=1e-5)
