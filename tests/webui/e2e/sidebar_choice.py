"""Choosing the sidebar's shown files before a browser test opens a page."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlencode

from playwright.sync_api import Page, expect


def choose_files(page: Page, base_url: str, stems: Sequence[str]) -> None:
    """Choose the sidebar's shown files through the server, as a checkbox would.

    Parameters
    ----------
    page : Page
        Browser page whose request context posts the choice.
    base_url : str
        Base URL of the running server.
    stems : Sequence[str]
        Shown files of the run in use. They are sent with the run of the
        sidebar, as its form does.
    """
    sidebar = page.request.get(f"{base_url}/sidebar/selection").text()
    match = re.search(r'<input type="hidden" name="run" value="([^"]+)">', sidebar)
    assert match is not None
    response = page.request.post(
        f"{base_url}/sidebar/selection/files",
        data=urlencode([("run", match.group(1)), *(("file", stem) for stem in stems)]),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.ok


def wait_for_sidebar(page: Page) -> None:
    """Wait until htmx has loaded and processed the sidebar's choices.

    Parameters
    ----------
    page : Page
        Browser page showing a screen.
    """
    expect(page.locator("#view-selection")).to_have_attribute("data-ready", "true")
