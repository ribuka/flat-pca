"""Browser tests of resizing the data table's columns on the data selection page."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
from playwright.sync_api import Locator, Page, expect

from flat_pca.webui.datatable import TableConfig
from flat_pca.webui.routes import catalog
from flat_pca.webui.services.file_table import file_table_config
from flat_pca.webui.settings import MetadataColumnSettings

pytestmark = pytest.mark.e2e

STORAGE_KEY = "datatable:column-widths:files"


@pytest.fixture(autouse=True)
def paged_file_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show two files per page, so that the table has a next page."""

    def paged(columns: Mapping[str, MetadataColumnSettings]) -> TableConfig:
        """Return the file table settings with a small page."""
        return replace(file_table_config(columns), page_size=2)

    monkeypatch.setattr(catalog, "file_table_config", paged)


def _open(page: Page, url: str) -> None:
    """Open the data selection page and wait for the first page of files."""
    page.goto(url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2"])


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order."""
    return page.locator("#files tbody tr td:nth-child(2)")


def _handle(page: Page, column: str) -> Locator:
    """Return the resize handle on the right edge of a column's header."""
    return page.locator(f'#files [data-dt-resize="{column}"]')


def _header_width(page: Page, column: str) -> float:
    """Return the width of a column's header cell in CSS pixels."""
    return _handle(page, column).evaluate("handle => handle.parentElement.getBoundingClientRect().width")


def _header_widths(page: Page) -> list[float]:
    """Return the widths of every header cell, the checkbox column first."""
    return page.locator("#files thead th").evaluate_all(
        "cells => cells.map((cell) => cell.getBoundingClientRect().width)"
    )


def _css_pixels(page: Page, value: str) -> float:
    """Return a CSS length evaluated in the file table, in pixels."""
    return page.locator("#files").evaluate(
        """(root, value) => {
          const probe = document.createElement("div");
          probe.style.width = value;
          root.append(probe);
          const width = probe.getBoundingClientRect().width;
          probe.remove();
          return width;
        }""",
        value,
    )


def _drag(page: Page, column: str, offset: float) -> None:
    """Drag a column's resize handle sideways by ``offset`` pixels."""
    handle = _handle(page, column)
    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    assert box is not None
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + offset, y, steps=5)
    page.mouse.up()


def _stored_widths(page: Page) -> object:
    """Return the column widths remembered for the file table, or None."""
    return page.evaluate(f"JSON.parse(localStorage.getItem('{STORAGE_KEY}'))")


def _open_menus(page: Page) -> Locator:
    """Return the open menus of the page."""
    return page.locator("[data-dt-menu]:popover-open")


def test_columns_start_at_one_width(page: Page, cataloged_server_url: str) -> None:
    """Every column but the checkbox column starts at --dt-col-width, whatever its values."""
    _open(page, cataloged_server_url)

    check, *columns = _header_widths(page)

    assert check == pytest.approx(_css_pixels(page, "var(--dt-check-width)"), abs=0.5)
    assert len(columns) > 2
    for width in columns:
        assert width == pytest.approx(_css_pixels(page, "var(--dt-col-width)"), abs=0.5)


def test_dragging_the_header_edge_resizes_the_column_and_is_remembered(
    page: Page, cataloged_server_url: str
) -> None:
    """A drag resizes the column live without opening its menu; the width holds across reloads."""
    _open(page, cataloged_server_url)
    handle = _handle(page, "n_rows")
    expect(handle).to_have_css("cursor", "col-resize")
    start = _header_width(page, "n_rows")
    others = _header_width(page, "n_steps")

    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    assert box is not None
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 40, y, steps=4)
    # The table follows the pointer before it is released.
    assert _header_width(page, "n_rows") == pytest.approx(start + 40, abs=1)
    expect(handle).to_have_attribute("data-dt-resizing", "")
    page.mouse.move(x + 80, y, steps=4)
    page.mouse.up()

    assert _header_width(page, "n_rows") == pytest.approx(start + 80, abs=1)
    assert _header_width(page, "n_steps") == pytest.approx(others, abs=0.5)
    expect(_open_menus(page)).to_have_count(0)
    assert _stored_widths(page) == {"n_rows": round(start + 80)}

    page.get_by_role("button", name="Next page").click()
    expect(_file_stems(page)).to_have_text(["run-10"])
    assert _header_width(page, "n_rows") == pytest.approx(start + 80, abs=1)

    page.locator("#files").get_by_role("searchbox", name="Search").fill("run-1")
    expect(_file_stems(page)).to_have_text(["run-1", "run-10"])
    assert _header_width(page, "n_rows") == pytest.approx(start + 80, abs=1)

    page.reload()
    _open(page, cataloged_server_url)
    assert _header_width(page, "n_rows") == pytest.approx(start + 80, abs=1)


def test_columns_never_get_narrower_than_the_minimum(
    page: Page, cataloged_server_url: str
) -> None:
    """Dragging far to the left stops at --dt-col-min-width; the histogram keeps its width."""
    _open(page, cataloged_server_url)
    minimum = _css_pixels(page, "var(--dt-col-min-width)")
    histogram = _css_pixels(page, "var(--dt-histogram-width)")
    assert minimum >= histogram

    _drag(page, "n_rows", -1000)

    assert _header_width(page, "n_rows") == pytest.approx(minimum, abs=1)
    rows_histogram = page.locator("#files th:has([data-dt-resize='n_rows']) [data-dt-histogram]")
    assert rows_histogram.evaluate("box => box.getBoundingClientRect().width") == pytest.approx(histogram, abs=0.5)
    sort = page.locator("#files [data-dt-column='n_rows']")
    assert sort.evaluate(
        "button => button.getBoundingClientRect().right <= button.closest('th').getBoundingClientRect().right"
    )


def test_double_click_fits_the_column_to_its_header_and_shown_values(
    page: Page, cataloged_server_url: str
) -> None:
    """A double-click widens a column to show a long value whole, and narrows one down to the minimum."""
    _open(page, cataloged_server_url)
    initial = _header_width(page, "n_rows")
    first_rows = page.locator("#files tbody tr:first-child td:nth-child(2) ~ td").last
    first_rows.evaluate("cell => { cell.textContent = '1234567890'.repeat(4); }")
    assert first_rows.evaluate("cell => cell.scrollWidth > cell.clientWidth")

    _handle(page, "n_rows").dblclick()

    assert _header_width(page, "n_rows") > initial
    assert first_rows.evaluate("cell => cell.scrollWidth <= cell.clientWidth")
    expect(_open_menus(page)).to_have_count(0)

    _handle(page, "n_steps").dblclick()

    minimum = _css_pixels(page, "var(--dt-col-min-width)")
    assert _header_width(page, "n_steps") == pytest.approx(minimum, abs=1)
    stored = _stored_widths(page)
    assert isinstance(stored, dict)
    assert set(stored) == {"n_rows", "n_steps"}


def test_reset_column_widths_returns_every_column_to_the_initial_width(
    page: Page, cataloged_server_url: str
) -> None:
    """"Reset column widths" in the "Columns" menu works only while a column was resized."""
    _open(page, cataloged_server_url)
    initial = _header_widths(page)
    table = page.locator("#files")
    reset = table.get_by_role("button", name="Reset column widths")
    table.get_by_role("button", name="Columns", exact=True).click()
    expect(reset).to_be_disabled()
    page.keyboard.press("Escape")

    _drag(page, "stem", 60)
    _drag(page, "n_rows", 30)
    assert _header_widths(page) != initial

    table.get_by_role("button", name="Columns", exact=True).click()
    expect(reset).to_be_enabled()
    reset.click()

    assert _header_widths(page) == pytest.approx(initial, abs=0.5)
    expect(reset).to_be_disabled()
    assert _stored_widths(page) == {}
    page.reload()
    _open(page, cataloged_server_url)
    assert _header_widths(page) == pytest.approx(initial, abs=0.5)


def test_resized_pinned_column_stays_in_place(page: Page, cataloged_server_url: str) -> None:
    """A resized pinned first column starts after the checkbox column, and the next column after it."""
    _open(page, cataloged_server_url)
    page.add_style_tag(content="#files .dt-scroll { max-width: 480px; }")
    page.locator("#files").get_by_role("button", name="Pin columns", exact=True).click()
    expect(page.locator("#files")).to_have_attribute("data-dt-pinned", "")

    _drag(page, "stem", 70)

    edges = page.evaluate(
        """() => {
          const box = document.querySelector("#files .dt-scroll");
          const cells = [...box.querySelectorAll("tbody tr:first-child > td")].slice(0, 3);
          const rects = () => cells.map((cell) => cell.getBoundingClientRect()).map((r) => [r.left, r.right]);
          box.scrollLeft = 0;
          const before = rects();
          box.scrollLeft = box.scrollWidth;
          const after = rects();
          box.scrollLeft = 0;
          return [before, after];
        }"""
    )
    (check, stem, nxt), (check_after, stem_after, nxt_after) = edges
    assert stem[0] == pytest.approx(check[1], abs=0.5)
    assert nxt[0] == pytest.approx(stem[1], abs=0.5)
    assert stem[1] - stem[0] == pytest.approx(_header_width(page, "stem"), abs=0.5)
    assert check_after == check
    assert stem_after == stem
    assert nxt_after[0] < nxt[0]


def test_column_widths_change_from_the_keyboard(page: Page, cataloged_server_url: str) -> None:
    """←/→ on a focused handle narrow or widen its column and remember it."""
    _open(page, cataloged_server_url)
    start = _header_width(page, "n_rows")
    handle = _handle(page, "n_rows")
    expect(handle).to_have_attribute("role", "separator")
    expect(handle).to_have_accessible_name("Resize column rows")

    handle.focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowLeft")

    assert _header_width(page, "n_rows") == pytest.approx(start + 10, abs=1)
    expect(handle).to_have_attribute("aria-valuenow", str(round(start + 10)))
    assert _stored_widths(page) == {"n_rows": round(start + 10)}


def test_cut_values_show_whole_as_a_tooltip(page: Page, cataloged_server_url: str) -> None:
    """A value cut short by its column ends in an ellipsis and shows whole on hover."""
    _open(page, cataloged_server_url)
    cell = page.locator("#files tbody tr:first-child td").last
    text = "9876543210" * 5
    cell.evaluate("(cell, text) => { cell.textContent = text; }", text)
    expect(cell).to_have_css("text-overflow", "ellipsis")

    cell.hover()
    expect(cell).to_have_attribute("title", text)

    file_cell = _file_stems(page).first
    path = file_cell.get_attribute("title")
    assert path is not None
    file_cell.hover()
    expect(file_cell).to_have_attribute("title", path)
