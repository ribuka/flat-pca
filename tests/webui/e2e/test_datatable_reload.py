"""Browser tests of how a data table reloads: the swapped parts, the focus, the page size, and the busy state."""

from __future__ import annotations

import re

import pytest
from datatable_menu import apply_column_menu, open_column_menu
from playwright.sync_api import Locator, Page, Route, expect

from flat_pca.webui.services import file_table

pytestmark = pytest.mark.e2e


def _table(page: Page) -> Locator:
    """Return the file table's container."""
    return page.locator("#files")


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order."""
    return page.locator("#files tbody tr td:nth-child(2)")


def _mark(page: Page, selector: str) -> None:
    """Mark the element found by a selector, to tell later whether it was replaced."""
    page.locator(selector).first.evaluate("element => { element.dataset.kept = 'yes'; }")


def test_reloads_swap_only_the_parts_that_change(page: Page, cataloged_server_url: str) -> None:
    """Sorting and filtering replace the rows and marks but keep the toolbar, menus, and headers."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    kept = [
        "#files .dt-toolbar",
        "#files .dt-search",
        '#files [data-dt-menu="lot"]',
        "#files thead th:nth-child(2)",
        "#files thead",
    ]
    for selector in kept:
        _mark(page, selector)
    _mark(page, "#files tbody")
    _mark(page, "#files .dt-footer")

    open_column_menu(table, "file").get_by_role("button", name="Desc").click()

    expect(stems).to_have_text(["run-10", "run-2", "run-1"])
    for selector in kept:
        expect(page.locator(selector).first).to_have_attribute("data-kept", "yes")
    expect(page.locator("#files tbody")).not_to_have_attribute("data-kept", "yes")
    expect(page.locator("#files .dt-footer")).not_to_have_attribute("data-kept", "yes")
    expect(page.locator("#files thead th:nth-child(2)")).to_have_attribute("aria-sort", "descending")
    expect(page.locator('#files [data-dt-column="stem"] .dt-sort-mark')).to_have_text("▼")

    lot = open_column_menu(table, "lot")
    lot.get_by_role("checkbox", name="A 1", exact=True).check()
    apply_column_menu(lot)

    expect(stems).to_have_text(["run-1"])
    expect(page.locator('#files [data-dt-menu="lot"]')).to_have_attribute("data-kept", "yes")
    expect(table.locator("[data-dt-filtered]")).to_have_count(1)
    expect(table.get_by_role("button", name="Remove filter lot ∈ {A}")).to_be_visible()
    expect(table.locator(".dt-top-value.dt-out")).to_have_text([r"B 1"], use_inner_text=True)
    expect(page.locator("[data-dt-page-range]")).to_have_text("1–1 of 1")
    # The header checkbox selects the rows matching the filter as swapped in.
    table.locator("[data-dt-check-all]").check()
    expect(page.locator("#files-selection")).to_have_value('["run-1"]')


def test_paging_keeps_the_focus_on_the_page_button(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page button pressed from the keyboard keeps the focus after the rows change."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 1)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1"])

    page.get_by_role("button", name="Next page").focus()
    page.keyboard.press("Enter")

    expect(stems).to_have_text(["run-2"])
    expect(page.get_by_role("button", name="Next page")).to_be_focused()
    page.keyboard.press("Enter")
    expect(stems).to_have_text(["run-10"])


def test_rows_per_page_keep_the_first_row_shown(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing the rows per page reloads the page that holds the first row shown."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 1)
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZES", (2, 50))
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    rows_per_page = page.get_by_label("Rows per page")
    expect(rows_per_page).to_have_value("1")
    page.get_by_role("button", name="Last page").click()
    expect(stems).to_have_text(["run-10"])

    rows_per_page.select_option("2")

    expect(stems).to_have_text(["run-10"])
    expect(page.locator("[data-dt-page-range]")).to_have_text("3–3 of 3")
    expect(rows_per_page).to_have_value("2")
    page.get_by_role("button", name="First page").click()
    expect(stems).to_have_text(["run-1", "run-2"])
    page.get_by_label("Rows per page").select_option("50")
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])


def test_reload_shows_that_the_table_is_busy(page: Page, cataloged_server_url: str) -> None:
    """While the rows are on their way, the table is busy and shows the chosen sort."""
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    table = _table(page)
    held: list[Route] = []

    def hold(route: Route) -> None:
        """Keep the request of the reload unanswered until the test lets it go."""
        held.append(route)

    page.route("**/catalog/files?*", hold)

    open_column_menu(table, "yield_pct").get_by_role("button", name="Desc").click()

    expect(table).to_have_attribute("aria-busy", "true")
    expect(table).to_have_class(re.compile(r"(^| )htmx-request( |$)"))
    expect(table.locator('[data-dt-column="yield_pct"] .dt-sort-mark')).to_have_text("▼")
    expect(table.locator('[data-dt-column="stem"] .dt-sort-mark')).to_have_text("")
    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    held[0].continue_()
    page.unroute("**/catalog/files?*", hold)

    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    expect(table).not_to_have_attribute("aria-busy", "true")
    expect(page.locator("#files thead th:nth-child(5)")).to_have_attribute("aria-sort", "descending")
