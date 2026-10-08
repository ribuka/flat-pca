"""Tests for the main part of the screens showing the sidebar's run and files."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from flat_pca.webui.app import create_app
from flat_pca.webui.settings import Settings

# The sidebar choices that replace the main part of each screen.
VIEW_SWAP = {
    "/": "",
    "/fit": "",
    "/transform": "run",
    "/model": "run",
    "/explore": "run files",
    "/scores": "run files",
    "/monitoring": "run",
}
# The screens that render only their main part for an htmx request.
VIEW_PAGES = ("/transform", "/model", "/explore", "/scores", "/monitoring")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Run the application on an empty workspace."""
    with TestClient(create_app(settings)) as opened:
        yield opened


def _main(html: str) -> tuple[str, str]:
    """Return the opening tag and the contents of the page's main element."""
    match = re.search(r'(<main class="content"[^>]*>)(.*)</main>', html, re.DOTALL)
    assert match is not None
    return match.group(1), match.group(2)


@pytest.mark.parametrize("path", list(VIEW_SWAP))
def test_main_names_the_choices_that_replace_it(client: TestClient, path: str) -> None:
    """The main element of every screen names the choices that replace it."""
    tag, _ = _main(client.get(path).text)

    assert f'data-view-swap="{VIEW_SWAP[path]}"' in tag


@pytest.mark.parametrize("path", VIEW_PAGES)
def test_htmx_request_gets_the_main_part_only(client: TestClient, path: str) -> None:
    """An htmx request gets the contents of the main element of the full page."""
    full = client.get(path)
    part = client.get(path, headers={"HX-Request": "true"})

    assert part.status_code == 200
    assert part.text.strip() == _main(full.text)[1].strip()
    for response in (full, part):
        assert response.headers["Vary"] == "HX-Request"
