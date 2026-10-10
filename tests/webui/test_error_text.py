"""Tests for shortening the error messages shown in the Web UI."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from flat_pca.webui.app import create_app
from flat_pca.webui.error_text import ELLIPSIS, ERROR_TEXT_LIMIT, truncate_error
from flat_pca.webui.services.runs import get_run, insert_run, update_run
from flat_pca.webui.settings import Settings
from flat_pca.webui.templating import TEMPLATES_DIR, templates
from flat_pca.webui.workspace import CATALOG_JOB

LONG_ERROR = "metadata CSV key No repeats: [" + ", ".join(map(str, range(2000))) + "]"


@pytest.mark.parametrize("length", [0, 1, ERROR_TEXT_LIMIT - 1, ERROR_TEXT_LIMIT])
def test_short_error_is_kept(length: int) -> None:
    """An error of at most the limit is shown as it is."""
    message = "x" * length

    assert truncate_error(message) == message


def test_long_error_is_cut_with_an_ellipsis() -> None:
    """An error over the limit keeps its first characters and ends with ``…``."""
    shown = truncate_error(LONG_ERROR)

    assert shown == LONG_ERROR[:ERROR_TEXT_LIMIT].rstrip() + ELLIPSIS
    assert len(shown) <= ERROR_TEXT_LIMIT + len(ELLIPSIS)


def test_cut_drops_trailing_whitespace() -> None:
    """The ellipsis follows the last kept word, not a dangling space."""
    assert truncate_error("abc def", limit=4) == "abc" + ELLIPSIS


def test_limit_and_non_string_message() -> None:
    """The limit is configurable and the message is converted with ``str``."""
    assert truncate_error(123456, limit=3) == "123" + ELLIPSIS


def test_every_displayed_error_is_truncated() -> None:
    """Every error element in the templates passes its text through the filter."""
    pattern = re.compile(r'class="error"[^>]*>\{\{(.*?)\}\}')
    found = {
        path.relative_to(TEMPLATES_DIR).as_posix(): expression.strip()
        for path in TEMPLATES_DIR.rglob("*.html")
        for expression in pattern.findall(path.read_text(encoding="utf-8"))
    }

    assert {
        "macros/run_status.html",
        "pages/explore.html",
        "pages/model.html",
        "pages/monitoring.html",
        "pages/scores.html",
        "partials/catalog_status.html",
    } <= found.keys()
    assert {
        name: expression
        for name, expression in found.items()
        if not expression.endswith("|error_text")
    } == {}


def test_run_status_shows_a_truncated_error() -> None:
    """The run status macro shows a failed run's error cut to the limit."""
    template = templates.env.from_string(
        '{% from "macros/run_status.html" import run_status %}'
        '{{ run_status(status, "fit") }}'
    )
    run = {"run_id": "fit-1", "status": "failed", "error": LONG_ERROR}

    html = template.render(status={"run": run, "active": False})

    assert truncate_error(LONG_ERROR) in html
    assert LONG_ERROR not in html


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application, opening its workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


def test_catalog_status_truncates_the_error_but_keeps_the_run_record(
    client: TestClient, settings: Settings
) -> None:
    """The catalog status shows the cut error while the run keeps the full text."""
    database = client.app.state.workspace.database
    run_dir = settings.runs_dir / "catalog-1"
    run_dir.mkdir(parents=True)
    insert_run(database, "catalog-1", CATALOG_JOB, {}, run_dir)
    update_run(database, "catalog-1", status="failed", error=LONG_ERROR)

    html = client.get("/catalog/status").text

    assert f'<p class="error">{truncate_error(LONG_ERROR)}</p>' in html
    assert LONG_ERROR not in html
    run = get_run(database, "catalog-1")
    assert run is not None
    assert run["error"] == LONG_ERROR
