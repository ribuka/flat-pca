"""Browser tests of the data selection page."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from flat_pca.webui.routes import catalog as catalog_routes

pytestmark = pytest.mark.e2e

CATALOG_TIMEOUT_MS = 60_000


def _file_rows(page: Page) -> Locator:
    """Return the file table's body rows."""
    return page.locator("#file-table tbody tr")


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order.

    Assert on all of them at once with ``to_have_text([...])``: it retries
    until the whole list matches, so a table not yet swapped by htmx cannot
    satisfy it the way a row count could.
    """
    return page.locator("#file-table tbody tr td:nth-child(2)")


def test_file_list_is_shown(page: Page, cataloged_server_url: str) -> None:
    """The file table loads every cataloged file and its metadata."""
    page.goto(cataloged_server_url)

    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    expect(page.locator('#file-table tr[data-stem="run-1"]')).to_contain_text("A")


def test_metadata_filters_narrow_file_list(
    page: Page, cataloged_server_url: str
) -> None:
    """Category and numeric metadata filters reload the file table."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])

    page.locator('select[name="eq__lot"]').select_option("B")
    expect(stems).to_have_text(["run-2"])

    page.locator('select[name="eq__lot"]').select_option("")
    page.get_by_label("yield_pct lower").fill("90")
    expect(stems).to_have_text(["run-1"])

    page.get_by_label("yield_pct lower").fill("")
    page.get_by_label("Filter by file name").fill("run-1")
    expect(stems).to_have_text(["run-1", "run-10"])
    expect(page.get_by_label("Filter by file name")).to_be_focused()


def test_column_names_sort_file_list(page: Page, cataloged_server_url: str) -> None:
    """Clicking a column name sorts ascending, and again descending."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    header = page.locator("#file-table th", has=page.locator('[data-sort="yield_pct"]'))

    page.locator('[data-sort="yield_pct"]').click()
    expect(stems).to_have_text(["run-2", "run-1", "run-10"])
    expect(header).to_have_attribute("aria-sort", "ascending")
    expect(header.locator(".sort-mark")).to_have_text("▲")

    page.locator('[data-sort="yield_pct"]').click()
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(header).to_have_attribute("aria-sort", "descending")
    expect(header.locator(".sort-mark")).to_have_text("▼")

    page.locator('[data-sort="stem"]').click()
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(header).not_to_have_attribute("aria-sort", re.compile(".*"))


def test_pages_keep_selection_and_select_saves_all(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selections on other pages are kept and saved together by Select."""
    monkeypatch.setattr(catalog_routes, "FILE_PAGE_SIZE", 2)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    expect(page.locator("[data-page-range]")).to_have_text("1–2 of 3")
    header = page.get_by_label("Select all filtered files")

    page.get_by_label("Select run-1").check()
    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-10"])
    expect(page.locator("[data-page-range]")).to_have_text("3–3 of 3")
    expect(header).to_have_js_property("indeterminate", True)
    page.get_by_label("Select run-10").check()

    page.get_by_role("button", name="1", exact=True).click()
    expect(stems).to_have_text(["run-1", "run-2"])
    expect(page.get_by_label("Select run-1")).to_be_checked()
    expect(page.get_by_label("Select run-2")).not_to_be_checked()

    page.get_by_role("button", name="Select").click()
    expect(page.locator("#selection-summary")).to_have_attribute(
        "data-selected-count", "2"
    )
    expect(page.locator('[data-sidebar="selection"]')).to_have_text("2 files")


def test_sorting_and_filtering_return_to_first_page(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the sort order or a filter shows the first page."""
    monkeypatch.setattr(catalog_routes, "FILE_PAGE_SIZE", 2)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])

    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-10"])
    page.locator('[data-sort="stem"]').click()
    expect(stems).to_have_text(["run-10", "run-2"])

    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-1"])
    page.get_by_label("Filter by file name").fill("run")
    expect(stems).to_have_text(["run-10", "run-2"])


def test_catalog_update_shows_progress_and_replaces_category_filters(
    page: Page, server_url: str
) -> None:
    """A catalog update polls its status, then reloads filters and files."""
    page.goto(server_url)
    expect(page.locator("#catalog-status")).to_contain_text(
        "No catalog has been built yet"
    )
    lot_options = page.locator('#file-table select[name="eq__lot"] option')
    expect(lot_options).to_have_text(["(all)"])
    expect(_file_rows(page)).to_have_count(0)

    page.get_by_role("button", name="Update catalog").click()

    status = page.locator("#catalog-status [data-run-status]")
    expect(status).to_have_attribute("data-run-status", re.compile("queued|running"))
    expect(page.locator("#catalog-status")).to_have_attribute(
        "hx-trigger", "every 1s"
    )
    expect(status).to_have_attribute(
        "data-run-status", "succeeded", timeout=CATALOG_TIMEOUT_MS
    )
    expect(lot_options).to_have_text(["(all)", "A", "B"])
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    expect(page.locator('[data-sidebar="catalog"]')).to_contain_text("succeeded")


def test_selecting_files_shows_success_icon_offline(
    page: Page, cataloged_server_url: str
) -> None:
    """Selecting files shows a green check icon from the bundled font.

    Requests to other hosts are aborted, so the icon must render without a
    network.
    """
    page.route(
        re.compile(r"^(?!" + re.escape(cataloged_server_url) + r")"),
        lambda route: route.abort(),
    )
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    icon = page.locator("#selection-summary .material-symbols-outlined")
    expect(icon).to_have_count(0)

    page.get_by_label("Select all filtered files").check()
    page.get_by_role("button", name="Select").click()

    expect(icon).to_have_text("check_circle")
    expect(icon).to_have_css("color", "rgb(26, 127, 55)")
    # The font starts loading when the icon appears.
    assert page.evaluate(
        "document.fonts.ready.then(() =>"
        " document.fonts.check('24px \"Material Symbols Outlined\"', 'check_circle'))"
    )
    # A ligature renders one square glyph instead of the text "check_circle".
    box = icon.bounding_box()
    assert box is not None
    assert box["width"] < box["height"] * 1.5


def test_header_checkbox_toggles_matching_rows(
    page: Page, cataloged_server_url: str
) -> None:
    """The header checkbox checks or unchecks every matching row and shows a mix."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    header = page.get_by_label("Select all filtered files")
    rows = page.locator("#file-table tbody input[data-stem-check]")
    expect(header).not_to_be_checked()

    header.check()
    for index in range(3):
        expect(rows.nth(index)).to_be_checked()

    rows.nth(1).uncheck()
    expect(header).not_to_be_checked()
    expect(header).to_have_js_property("indeterminate", True)

    header.click()
    for index in range(3):
        expect(rows.nth(index)).to_be_checked()
    expect(header).to_have_js_property("indeterminate", False)

    header.uncheck()
    for index in range(3):
        expect(rows.nth(index)).not_to_be_checked()

    # The header follows the rows matching the filters; it leaves the
    # selection of the other rows alone.
    rows.nth(0).check()
    page.locator('select[name="eq__lot"]').select_option("B")
    expect(_file_stems(page)).to_have_text(["run-2"])
    expect(header).not_to_be_checked()
    expect(header).to_have_js_property("indeterminate", False)
    header.check()
    expect(rows.nth(0)).to_be_checked()

    page.locator('select[name="eq__lot"]').select_option("")
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    expect(rows.nth(0)).to_be_checked()
    expect(rows.nth(1)).to_be_checked()
    expect(rows.nth(2)).not_to_be_checked()
    expect(header).to_have_js_property("indeterminate", True)


def test_file_table_scrolls_under_a_fixed_header(
    page: Page, cataloged_server_url: str
) -> None:
    """The table has a height limit and keeps its header on top."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])

    expect(page.locator(".file-table-scroll")).to_have_css("overflow", "auto")
    assert page.locator(".file-table-scroll").evaluate(
        "box => getComputedStyle(box).maxHeight !== 'none'"
    )
    expect(page.locator(".file-table thead th").first).to_have_css("position", "sticky")


def test_groups_collapse(page: Page, cataloged_server_url: str) -> None:
    """Clicking a group's heading folds its body away and back."""
    page.goto(cataloged_server_url)
    files = page.locator('details[data-group="files"]')
    name_filter = files.get_by_placeholder("contains")
    expect(name_filter).to_be_visible()

    files.locator(":scope > summary").click()
    expect(name_filter).to_be_hidden()

    files.locator(":scope > summary").click()
    expect(name_filter).to_be_visible()
