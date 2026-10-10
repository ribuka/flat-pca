"""Browser tests of the data table's "Pin columns" toggle on the data selection page."""

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

STORAGE_KEY = "datatable:pinned-columns:files"


@pytest.fixture(autouse=True)
def paged_file_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show two files per page, so that the table has a next page."""

    def paged(columns: Mapping[str, MetadataColumnSettings]) -> TableConfig:
        """Return the file table settings with a small page."""
        return replace(file_table_config(columns), page_size=2)

    monkeypatch.setattr(catalog, "file_table_config", paged)


def _table(page: Page) -> Locator:
    """Return the file table's container."""
    return page.locator("#files")


def _file_stems(page: Page) -> Locator:
    """Return the file name cells of the file table, in display order."""
    return page.locator("#files tbody tr td:nth-child(2)")


def _toggle(page: Page) -> Locator:
    """Return the "Pin columns" toggle of the file table."""
    return _table(page).get_by_role("button", name="Pin columns", exact=True)


def _narrow(page: Page) -> None:
    """Make the table's scroll box narrow, as on a narrow window, across reloads of the table."""
    page.add_style_tag(content="#files .dt-scroll { max-width: 240px; }")


def _first_cells_after_scrolling(page: Page) -> tuple[list[float], list[float]]:
    """Scroll the first row sideways and return where its first three cells were and are.

    Returns
    -------
    tuple[list[float], list[float]]
        Left edges of the checkbox, file name, and next cells before and
        after scrolling to the right end; the scroll position is reset
        afterwards.
    """
    return page.evaluate(
        """() => {
          const box = document.querySelector("#files .dt-scroll");
          const cells = [...box.querySelectorAll("tbody tr:first-child > td")].slice(0, 3);
          const lefts = () => cells.map((cell) => cell.getBoundingClientRect().left);
          box.scrollLeft = 0;
          const before = lefts();
          box.scrollLeft = box.scrollWidth;
          const after = lefts();
          box.scrollLeft = 0;
          return [before, after];
        }"""
    )


def _expect_pinned(page: Page, pinned: bool) -> None:
    """Check the toggle's state, and whether the left cells stay while the table scrolls sideways."""
    expect(_toggle(page)).to_have_attribute("aria-pressed", "true" if pinned else "false")
    if pinned:
        expect(_table(page)).to_have_attribute("data-dt-pinned", "")
    else:
        expect(_table(page)).not_to_have_attribute("data-dt-pinned", "")
    scroll = page.locator("#files .dt-scroll")
    assert scroll.evaluate("box => box.scrollWidth > box.clientWidth")
    before, after = _first_cells_after_scrolling(page)
    assert after[2] < before[2]
    if pinned:
        assert after[:2] == before[:2]
    else:
        assert after[0] < before[0]
        assert after[1] < before[1]


def test_pin_toggle_pins_the_left_columns_and_remembers_it(
    page: Page, cataloged_server_url: str
) -> None:
    """Pinning is off at first; the toggle's choice holds across paging, filtering, and reloads."""
    page.goto(cataloged_server_url)
    _narrow(page)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    _expect_pinned(page, False)

    _toggle(page).click()
    _expect_pinned(page, True)
    assert page.evaluate(f"localStorage.getItem('{STORAGE_KEY}')") == "true"

    page.get_by_role("button", name="Next page").click()
    expect(stems).to_have_text(["run-10"])
    _expect_pinned(page, True)

    _table(page).get_by_role("searchbox", name="Search").fill("run-1")
    expect(stems).to_have_text(["run-1", "run-10"])
    _expect_pinned(page, True)

    # Reloading the page starts the table over, still pinned.
    page.reload()
    _narrow(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    _expect_pinned(page, True)

    _toggle(page).click()
    _expect_pinned(page, False)
    assert page.evaluate(f"localStorage.getItem('{STORAGE_KEY}')") == "false"
    page.reload()
    _narrow(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    _expect_pinned(page, False)


def test_pinned_cells_cover_the_cells_scrolled_under_them(
    page: Page, cataloged_server_url: str
) -> None:
    """The pinned file name cell is drawn above the cells scrolling under it; the header stays sticky."""
    page.goto(cataloged_server_url)
    _narrow(page)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2"])
    _toggle(page).click()
    expect(_table(page)).to_have_attribute("data-dt-pinned", "")

    page.locator("#files .dt-scroll").evaluate("box => { box.scrollLeft = box.scrollWidth; }")

    file_cell = _file_stems(page).first
    box = file_cell.bounding_box()
    assert box is not None
    assert file_cell.evaluate(
        "(cell, [x, y]) => cell.contains(document.elementFromPoint(x, y))",
        [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2],
    )
    expect(page.locator("#files thead th").nth(1)).to_have_css("position", "sticky")


def test_pin_toggle_works_from_the_keyboard(page: Page, cataloged_server_url: str) -> None:
    """Tab reaches the toggle; Space and Enter switch it."""
    page.goto(cataloged_server_url)
    expect(_file_stems(page)).to_have_text(["run-1", "run-2"])
    _table(page).get_by_role("button", name="Columns", exact=True).focus()
    page.keyboard.press("Tab")
    expect(_toggle(page)).to_be_focused()

    page.keyboard.press("Space")
    expect(_toggle(page)).to_have_attribute("aria-pressed", "true")
    expect(_table(page)).to_have_attribute("data-dt-pinned", "")
    page.keyboard.press("Enter")
    expect(_toggle(page)).to_have_attribute("aria-pressed", "false")
    expect(_table(page)).not_to_have_attribute("data-dt-pinned", "")


def test_pin_toggle_keeps_the_choice_without_storage(
    page: Page, cataloged_server_url: str
) -> None:
    """Where localStorage fails, the pinned columns stay pinned across reloads of the table."""
    page.add_init_script(
        "for (const name of ['getItem', 'setItem']) {"
        " Storage.prototype[name] = () => { throw new DOMException('denied', 'QuotaExceededError'); }; }"
    )
    page.goto(cataloged_server_url)
    _narrow(page)
    stems = _file_stems(page)
    expect(stems).to_have_text(["run-1", "run-2"])
    _expect_pinned(page, False)

    _toggle(page).click()
    _expect_pinned(page, True)
    page.get_by_role("button", name="Next page").click()
    expect(stems).to_have_text(["run-10"])
    _expect_pinned(page, True)
