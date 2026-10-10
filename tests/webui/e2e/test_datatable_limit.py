"""Browser tests of the data table's selection limit on the data selection page."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
from datatable_menu import close_column_menu, open_column_menu
from playwright.sync_api import Locator, Page, expect

from flat_pca.webui.datatable import TableConfig
from flat_pca.webui.routes import catalog
from flat_pca.webui.services.file_table import file_table_config
from flat_pca.webui.settings import MetadataColumnSettings

pytestmark = pytest.mark.e2e

MAX_SELECTED = 2


@pytest.fixture(autouse=True)
def limited_file_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limit the file table to ``MAX_SELECTED`` files, two rows per page."""

    def limited(columns: Mapping[str, MetadataColumnSettings]) -> TableConfig:
        """Return the file table settings with the limit and a small page."""
        return replace(file_table_config(columns), page_size=2, max_selected=MAX_SELECTED)

    monkeypatch.setattr(catalog, "file_table_config", limited)


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order."""
    return page.locator("#files tbody tr td:nth-child(2)")


def _check(page: Page, stem: str) -> Locator:
    """Return the row checkbox of the file ``stem``."""
    return page.get_by_label(f"Select {stem}", exact=True)


def _search(page: Page, text: str) -> None:
    """Search the file names from the file column's menu, then close it."""
    menu = open_column_menu(page.locator("#files"), "file")
    menu.get_by_label("Filter by file name").fill(text)
    close_column_menu(page, menu)


def _limit_notice(page: Page) -> Locator:
    """Return the notice that the selection limit is reached."""
    return page.locator("#files [data-dt-limit]")


def test_limit_disables_unselected_rows_across_pages_and_filters(
    page: Page, cataloged_server_url: str
) -> None:
    """At the limit only selected rows can be changed, after paging and filtering too."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    header = page.get_by_label("Select all filtered files")
    expect(_limit_notice(page)).to_be_hidden()
    # Selecting all three files would pass the limit.
    expect(header).to_be_disabled()

    _check(page, "run-1").check()
    expect(_check(page, "run-2")).to_be_enabled()
    _check(page, "run-2").check()
    expect(_limit_notice(page)).to_be_visible()
    expect(_limit_notice(page)).to_contain_text("The limit of 2 selected rows is reached")
    expect(_check(page, "run-1")).to_be_enabled()

    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-10"])
    expect(_check(page, "run-10")).to_be_disabled()
    expect(_limit_notice(page)).to_be_visible()

    _search(page, "run-1")
    expect(stems).to_have_text(["run-1", "run-10"])
    expect(_check(page, "run-1")).to_be_checked()
    expect(_check(page, "run-1")).to_be_enabled()
    expect(_check(page, "run-10")).to_be_disabled()

    _check(page, "run-1").uncheck()
    expect(_limit_notice(page)).to_be_hidden()
    expect(_check(page, "run-10")).to_be_enabled()
    # run-2 stays selected outside the filter, so run-1 and run-10 do not fit.
    expect(header).to_be_disabled()
    _check(page, "run-10").check()
    expect(_check(page, "run-1")).to_be_disabled()

    open_column_menu(page.locator("#files"), "file").get_by_role("button", name="Desc").click()
    expect(stems).to_have_text(["run-10", "run-1"])
    expect(_check(page, "run-1")).to_be_disabled()
    expect(_check(page, "run-10")).to_be_checked()
    assert sorted(page.evaluate("JSON.parse(document.getElementById('files-selection').value)")) == [
        "run-10",
        "run-2",
    ]


def test_header_checkbox_selects_matching_rows_within_the_limit(
    page: Page, cataloged_server_url: str
) -> None:
    """The header selects every matching row when they fit and clears them; events carry the limit."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2"])
    page.evaluate(
        "window.dtDetails = [];"
        " document.addEventListener('datatable:selection-change',"
        " (event) => window.dtDetails.push(event.detail));"
    )
    header = page.get_by_label("Select all filtered files")

    _search(page, "run-1")
    expect(_file_stems(page)).to_have_text(["run-1", "run-10"])
    expect(header).to_be_enabled()

    header.check()
    expect(_check(page, "run-1")).to_be_checked()
    expect(_check(page, "run-10")).to_be_checked()
    expect(_limit_notice(page)).to_be_visible()
    assert page.evaluate("window.dtDetails.at(-1)") == {
        "keys": ["run-1", "run-10"],
        "max": MAX_SELECTED,
    }

    _search(page, "run-2")
    expect(_file_stems(page)).to_have_text(["run-2"])
    expect(header).to_be_disabled()
    expect(_check(page, "run-2")).to_be_disabled()

    _search(page, "run-1")
    expect(_file_stems(page)).to_have_text(["run-1", "run-10"])
    expect(header).to_be_checked()
    expect(header).to_be_enabled()
    header.uncheck()
    expect(_check(page, "run-1")).not_to_be_checked()
    expect(_check(page, "run-10")).not_to_be_checked()
    expect(_limit_notice(page)).to_be_hidden()
    assert page.evaluate("window.dtDetails.at(-1)") == {"keys": [], "max": MAX_SELECTED}
