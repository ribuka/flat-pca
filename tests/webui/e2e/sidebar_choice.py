"""Choosing the sidebar's shown files before a browser test opens a page."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from playwright.sync_api import Locator, Page, expect


def choose_files(page: Page, base_url: str, stems: Sequence[str]) -> None:
    """Choose the sidebar's shown files through the server, as the dialog would.

    Parameters
    ----------
    page : Page
        Browser page whose request context posts the choice.
    base_url : str
        Base URL of the running server.
    stems : Sequence[str]
        Shown files of the run in use. They are sent with the run of the
        shown-file dialog, as its form does.
    """
    dialog = page.request.get(f"{base_url}/sidebar/selection/files/dialog").text()
    match = re.search(r'<input type="hidden" name="run" value="([^"]+)">', dialog)
    assert match is not None
    response = page.request.post(
        f"{base_url}/sidebar/selection/files",
        form={"run": match.group(1), "stems": json.dumps(list(stems))},
    )
    assert response.ok


def shown_files(page: Page) -> Locator:
    """Return the items of the sidebar's read-only list of shown files.

    Parameters
    ----------
    page : Page
        Browser page showing a screen.

    Returns
    -------
    Locator
        One ``li[data-stem]`` per shown file.
    """
    return page.locator("#view-selection [data-view-file-list] li[data-stem]")


def open_file_dialog(page: Page) -> Locator:
    """Open the shown-file dialog and wait until its table has loaded.

    Parameters
    ----------
    page : Page
        Browser page showing a screen with the sidebar loaded.

    Returns
    -------
    Locator
        The open dialog.
    """
    page.get_by_role("button", name="Choose shown files").click()
    dialog = page.locator("#view-files-dialog")
    expect(dialog).to_be_visible()
    expect(dialog.locator("tr[data-dt-key]").first).to_be_visible()
    return dialog


def select_in_dialog(page: Page, stems: Sequence[str]) -> None:
    """Toggle files in the shown-file dialog and save them with "Select".

    Parameters
    ----------
    page : Page
        Browser page marked by ``mark_page``; wait for the refresh with
        ``wait_for_view_refresh`` afterwards.
    stems : Sequence[str]
        Files whose row checkbox is clicked (checked or cleared).
    """
    dialog = open_file_dialog(page)
    for stem in stems:
        dialog.get_by_role("checkbox", name=f"Select {stem}", exact=True).click()
    dialog.get_by_role("button", name="Select", exact=True).click()
    expect(dialog).to_be_hidden()


def wait_for_sidebar(page: Page) -> None:
    """Wait until htmx has loaded and processed the sidebar's choices.

    Parameters
    ----------
    page : Page
        Browser page showing a screen.
    """
    expect(page.locator("#view-selection")).to_have_attribute("data-ready", "true")


def mark_page(page: Page) -> None:
    """Mark the loaded page, so that a later reload can be detected.

    Parameters
    ----------
    page : Page
        Browser page showing a screen.
    """
    page.evaluate("() => { window.flatPcaMarked = true; }")


def wait_for_view_refresh(page: Page) -> None:
    """Wait until the refresh after a choice of the run or files ends.

    The overlay blocks the page from the choice until the sidebar and the
    main part are replaced. The page must not have been reloaded since
    ``mark_page``.

    Parameters
    ----------
    page : Page
        Browser page marked by ``mark_page``.
    """
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")
    assert page.evaluate("() => window.flatPcaMarked === true")
