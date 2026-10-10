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


def _short_table(page: Page) -> None:
    """Make the file table's scroll box only a few rows high, so that its rows scroll."""
    page.add_style_tag(content="#files { --dt-max-height: 10rem; }")


def _scroll_top(page: Page) -> float:
    """Return how far the file table's scroll box is scrolled down."""
    return page.locator("#files .dt-scroll").evaluate("box => box.scrollTop")


def test_new_rows_show_from_the_top_of_the_table(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paging and sorting after scrolling down show the new rows from the first one."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 2)
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    _short_table(page)
    page.locator("#files .dt-scroll").evaluate("box => { box.scrollTop = box.scrollHeight; }")
    assert _scroll_top(page) > 0

    page.get_by_role("button", name="Next page").click()

    expect(stems).to_have_text(["run-10"])
    assert _scroll_top(page) == 0
    page.get_by_role("button", name="Previous page").click()
    expect(stems).to_have_text(["run-1", "run-2"])
    page.locator("#files .dt-scroll").evaluate("box => { box.scrollTop = box.scrollHeight; }")
    open_column_menu(_table(page), "file").get_by_role("button", name="Desc").click()
    expect(stems).to_have_text(["run-10", "run-2"])
    assert _scroll_top(page) == 0


def test_rows_per_page_show_the_first_row_at_the_top(
    page: Page, cataloged_server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After choosing more rows per page, the row that began the page is at the top of the table."""
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 1)
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZES", (3,))
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    _short_table(page)
    page.get_by_role("button", name="Next page").click()
    expect(stems).to_have_text(["run-2"])

    page.get_by_label("Rows per page").select_option("3")

    expect(stems).to_have_text(["run-1", "run-2", "run-10"])
    gap = page.locator("#files").evaluate(
        "root => root.querySelector('tr[data-dt-key=\"run-2\"]').getBoundingClientRect().top"
        " - root.querySelector('thead th').getBoundingClientRect().bottom"
    )
    assert abs(gap) <= 1
    assert _scroll_top(page) > 0


def test_a_failed_change_of_rows_per_page_keeps_no_position(
    page: Page,
    cataloged_server_url: str,
    monkeypatch: pytest.MonkeyPatch,
    expected_console_errors: list[str],
) -> None:
    """After a change of the rows per page fails, a sort still shows its rows from the first one."""
    expected_console_errors.extend(["500", "htmx:responseError"])
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZE", 1)
    monkeypatch.setattr(file_table, "FILE_PAGE_SIZES", (3,))
    page.goto(cataloged_server_url)
    stems = _file_stems(page)
    _short_table(page)
    page.get_by_role("button", name="Next page").click()
    expect(stems).to_have_text(["run-2"])
    failed: list[str] = []

    def fail(route: Route) -> None:
        """Answer the change of the rows per page with a server error."""
        failed.append(route.request.url)
        route.fulfill(status=500, body="boom")

    page.route("**/catalog/files?*page_size=3*", fail)
    page.get_by_label("Rows per page").select_option("3")
    expect(page.locator("#files")).not_to_have_attribute("aria-busy", "true")
    assert len(failed) == 1
    page.unroute("**/catalog/files?*page_size=3*", fail)

    open_column_menu(_table(page), "file").get_by_role("button", name="Desc").click()

    expect(stems).to_have_text(["run-10", "run-2", "run-1"])
    assert _scroll_top(page) == 0
