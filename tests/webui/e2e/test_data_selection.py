"""Browser tests of the data selection page."""

from __future__ import annotations

import re
from typing import Literal

import pytest
from datatable_menu import close_column_menu, column_menu, open_column_menu
from playwright.sync_api import Locator, Page, expect

from flat_pca.webui.services import file_table

pytestmark = pytest.mark.e2e

CATALOG_TIMEOUT_MS = 60_000


def _file_rows(page: Page) -> Locator:
    """Return the file table's body rows."""
    return page.locator("#files tbody tr")


def _table(page: Page) -> Locator:
    """Return the file table's container."""
    return page.locator("#files")


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order.

    Assert on all of them at once with ``to_have_text([...])``: it retries
    until the whole list matches, so a table not yet swapped by htmx cannot
    satisfy it the way a row count could.
    """
    return page.locator("#files tbody tr td:nth-child(2)")


def test_file_list_is_shown(page: Page, cataloged_server_url: str) -> None:
    """The file table loads every cataloged file and its metadata."""
    page.goto(cataloged_server_url)

    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    expect(page.locator('#files tr[data-dt-key="run-1"]')).to_contain_text("A")


def test_column_menus_filter_file_list(page: Page, cataloged_server_url: str) -> None:
    """Value, bound, null, and search filters in the column menus reload the file table."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    filtered = table.locator("[data-dt-filtered]")
    expect(filtered).to_have_count(0)

    lot = open_column_menu(table, "lot")
    lot.get_by_role("checkbox", name="B 1", exact=True).check()
    expect(stems).to_have_text(["run-2"])
    # The menu stays open across the reload, so a second value adds to the first.
    expect(lot).to_be_visible()
    lot.get_by_role("checkbox", name="A 1", exact=True).check()
    expect(stems).to_have_text(["run-1", "run-2"])
    expect(lot.get_by_role("checkbox", name="B 1", exact=True)).to_be_checked()
    expect(filtered).to_have_count(1)
    # A value filter drops the null values, so "Is null" leaves no row.
    lot.get_by_role("radio", name="Is null").check()
    expect(stems).to_have_text([])
    lot.get_by_role("checkbox", name="A 1", exact=True).uncheck()
    lot.get_by_role("checkbox", name="B 1", exact=True).uncheck()
    expect(stems).to_have_text(["run-10"])
    expect(lot.locator(".dt-nulls legend")).to_have_text("Null values 1")
    lot.get_by_role("radio", name="Any").check()
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(filtered).to_have_count(0)
    close_column_menu(page, lot)

    yield_pct = open_column_menu(table, "yield_pct")
    yield_pct.get_by_label("yield_pct lower").fill("90")
    expect(stems).to_have_text(["run-1"])
    expect(yield_pct.get_by_label("yield_pct lower")).to_be_focused()
    yield_pct.get_by_label("yield_pct lower").fill("")
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    yield_pct.get_by_role("radio", name="Is not null").check()
    expect(stems).to_have_text(["run-1", "run-2"])
    close_column_menu(page, yield_pct)
    expect(filtered).to_have_count(1)

    name = open_column_menu(table, "file")
    name.get_by_label("Filter by file name").fill("1 RUN")
    expect(stems).to_have_text(["run-1"])
    expect(name.get_by_label("Filter by file name")).to_be_focused()
    expect(name.get_by_label("Filter by file name")).to_have_value("1 RUN")
    expect(filtered).to_have_count(2)


def test_column_menu_sorts_file_list(page: Page, cataloged_server_url: str) -> None:
    """The menu sorts ascending or descending or clears the sort, and closes."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    header = table.locator("th", has=page.locator('[data-dt-column="yield_pct"]'))
    expect(header.locator(".dt-type")).to_have_text("f64")

    open_column_menu(table, "yield_pct").get_by_role("button", name="Asc").click()
    expect(stems).to_have_text(["run-2", "run-1", "run-10"])
    expect(column_menu(table, "yield_pct")).to_be_hidden()
    expect(header).to_have_attribute("aria-sort", "ascending")
    expect(header.locator(".dt-sort-mark")).to_have_text("▲")

    open_column_menu(table, "yield_pct").get_by_role("button", name="Desc").click()
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(header).to_have_attribute("aria-sort", "descending")
    expect(header.locator(".dt-sort-mark")).to_have_text("▼")

    menu = open_column_menu(table, "yield_pct")
    expect(menu.get_by_role("button", name="Desc")).to_have_attribute("aria-pressed", "true")
    menu.get_by_role("button", name="Clear sort").click()
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(header).not_to_have_attribute("aria-sort", re.compile(".*"))
    expect(header.locator(".dt-sort-mark")).to_have_text("")


def test_column_menu_works_from_the_keyboard(page: Page, cataloged_server_url: str) -> None:
    """A column name opens its menu with Enter, Tab reaches it, and Escape closes it."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    button = table.get_by_role("button", name="lot", exact=True)
    menu = column_menu(table, "lot")

    button.focus()
    page.keyboard.press("Enter")
    expect(menu).to_be_visible()
    page.keyboard.press("Tab")
    expect(menu.get_by_role("button", name="Asc")).to_be_focused()
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    expect(button).to_be_focused()

    page.keyboard.press("Enter")
    expect(menu).to_be_visible()
    page.mouse.click(5, 5)
    expect(menu).to_be_hidden()

    # A menu reopened after a reload also returns the focus on Escape.
    button.focus()
    page.keyboard.press("Enter")
    # Asc, Desc (Clear sort is disabled while not sorted), then the first value.
    for _ in range(3):
        page.keyboard.press("Tab")
    expect(menu.get_by_role("checkbox", name="A 1", exact=True)).to_be_focused()
    page.keyboard.press("Space")
    expect(_file_stems(page)).to_have_text(["run-1"])
    expect(menu.get_by_role("checkbox", name="A 1", exact=True)).to_be_focused()
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    expect(table.get_by_role("button", name="lot", exact=True)).to_be_focused()


def test_column_menu_copies_the_column_name(page: Page, cataloged_server_url: str) -> None:
    """The copy button puts the frame's column name on the clipboard and closes the menu."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)

    open_column_menu(table, "file").get_by_role("button", name="Copy column name").click()

    expect(column_menu(table, "file")).to_be_hidden()
    assert page.evaluate("navigator.clipboard.readText()") == "stem"


def test_column_menu_is_not_clipped_by_the_table(page: Page, cataloged_server_url: str) -> None:
    """A menu taller than the table's scroll box shows whole, over the page."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    scroll = table.locator(".dt-scroll")
    scroll.evaluate("box => { box.style.maxHeight = '4rem'; }")

    menu = open_column_menu(table, "lot")

    menu_box = menu.bounding_box()
    scroll_box = scroll.bounding_box()
    assert menu_box is not None and scroll_box is not None
    assert menu_box["height"] > scroll_box["height"]
    # Its top and bottom are on top of the page, not clipped or covered.
    corners = [
        [menu_box["x"] + 10, menu_box["y"] + 5],
        [menu_box["x"] + 10, menu_box["y"] + menu_box["height"] - 5],
    ]
    assert menu.evaluate(
        "(menu, corners) => corners.every(([x, y]) => menu.contains(document.elementFromPoint(x, y)))",
        corners,
    )


def test_pages_keep_selection_and_select_saves_all(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selections on other pages are kept and saved together by Select."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 2)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    expect(page.locator("[data-dt-page-range]")).to_have_text("1–2 of 3")
    header = page.get_by_label("Select all filtered files")

    page.get_by_label("Select run-1", exact=True).check()
    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-10"])
    expect(page.locator("[data-dt-page-range]")).to_have_text("3–3 of 3")
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
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 2)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])

    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-10"])
    open_column_menu(_table(page), "file").get_by_role("button", name="Desc").click()
    expect(stems).to_have_text(["run-10", "run-2"])

    page.get_by_role("button", name="Next").click()
    expect(stems).to_have_text(["run-1"])
    open_column_menu(_table(page), "file").get_by_label("Filter by file name").fill("run")
    expect(stems).to_have_text(["run-10", "run-2"])


def test_catalog_update_shows_progress_and_replaces_category_filters(
    page: Page, server_url: str
) -> None:
    """A catalog update polls its status, then reloads filters and files."""
    page.goto(server_url)
    expect(page.locator("#catalog-status")).to_contain_text(
        "No catalog has been built yet"
    )
    lot_options = page.locator('#files input[name="files.eq__lot"]')
    expect(lot_options).to_have_count(0)
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
    expect(lot_options).to_have_count(2)
    expect(lot_options.nth(0)).to_have_value("A")
    expect(lot_options.nth(1)).to_have_value("B")
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
    rows = page.locator("#files tbody input[data-dt-row-check]")
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
    table = _table(page)
    lot = open_column_menu(table, "lot")
    lot.get_by_role("checkbox", name="B 1", exact=True).check()
    expect(_file_stems(page)).to_have_text(["run-2"])
    close_column_menu(page, lot)
    expect(header).not_to_be_checked()
    expect(header).to_have_js_property("indeterminate", False)
    header.check()
    expect(rows.nth(0)).to_be_checked()

    lot = open_column_menu(table, "lot")
    lot.get_by_role("checkbox", name="B 1", exact=True).uncheck()
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    close_column_menu(page, lot)
    expect(rows.nth(0)).to_be_checked()
    expect(rows.nth(1)).to_be_checked()
    expect(rows.nth(2)).not_to_be_checked()
    expect(header).to_have_js_property("indeterminate", True)


@pytest.mark.parametrize(
    ("header", "role", "name", "expected"),
    [
        ("lot", "checkbox", "B 1", ["run-1", "run-2"]),
        ("lot", "radio", "Is null", ["run-1", "run-10"]),
        ("yield_pct", "radio", "Is null", ["run-1", "run-10"]),
    ],
)
def test_header_checkbox_selects_the_rows_of_each_filter(
    page: Page,
    cataloged_server_url: str,
    header: str,
    role: Literal["checkbox", "radio"],
    name: str,
    expected: list[str],
) -> None:
    """With a value or null filter, the header adds exactly the matching rows."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    page.get_by_label("Select run-1", exact=True).check()

    menu = open_column_menu(_table(page), header)
    menu.get_by_role(role, name=name, exact=True).check()
    expect(_file_stems(page)).not_to_have_text(["run-1", "run-2", "run-10"])
    close_column_menu(page, menu)
    page.get_by_label("Select all filtered files").check()

    selection = page.evaluate("JSON.parse(document.getElementById('files-selection').value)")
    assert sorted(selection) == sorted(expected)


def test_header_checkbox_selects_the_rows_of_a_search(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search in any word order selects its rows with the header, on every page."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 1)
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1"])

    menu = open_column_menu(_table(page), "file")
    menu.get_by_label("Filter by file name").fill("1 RUN-")
    expect(page.locator("[data-dt-page-range]")).to_have_text("1–1 of 2")
    close_column_menu(page, menu)
    page.get_by_label("Select all filtered files").check()

    selection = page.evaluate("JSON.parse(document.getElementById('files-selection').value)")
    assert selection == ["run-1", "run-10"]


def test_file_table_scrolls_under_a_fixed_header(
    page: Page, cataloged_server_url: str
) -> None:
    """The table has a height limit and keeps its header on top."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])

    expect(page.locator(".dt-scroll")).to_have_css("overflow", "auto")
    assert page.locator(".dt-scroll").evaluate(
        "box => getComputedStyle(box).maxHeight !== 'none'"
    )
    expect(page.locator(".dt-table thead th").first).to_have_css("position", "sticky")


def test_groups_collapse(page: Page, cataloged_server_url: str) -> None:
    """Clicking a group's heading folds its body away and back."""
    page.goto(cataloged_server_url)
    files = page.locator('details[data-group="files"]')
    name_column = files.get_by_role("button", name="file", exact=True)
    expect(name_column).to_be_visible()

    files.locator(":scope > summary").click()
    expect(name_column).to_be_hidden()

    files.locator(":scope > summary").click()
    expect(name_column).to_be_visible()
