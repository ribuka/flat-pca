"""Browser tests of the data selection page."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

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
    page.get_by_label("yield_pct 下限").fill("90")
    expect(stems).to_have_text(["run-1"])


def test_catalog_update_shows_progress_and_replaces_category_filters(
    page: Page, server_url: str
) -> None:
    """A catalog update polls its status, then reloads filters and files."""
    page.goto(server_url)
    expect(page.locator("#catalog-status")).to_contain_text(
        "catalog はまだ作成されていません"
    )
    lot_options = page.locator('#category-filters select[name="eq__lot"] option')
    expect(lot_options).to_have_text(["(すべて)"])
    expect(_file_rows(page)).to_have_count(0)

    page.get_by_role("button", name="catalog 更新").click()

    status = page.locator("#catalog-status [data-run-status]")
    expect(status).to_have_attribute("data-run-status", re.compile("queued|running"))
    expect(page.locator("#catalog-status")).to_have_attribute(
        "hx-trigger", "every 1s"
    )
    expect(status).to_have_attribute(
        "data-run-status", "succeeded", timeout=CATALOG_TIMEOUT_MS
    )
    expect(lot_options).to_have_text(["(すべて)", "A", "B"])
    expect(_file_stems(page)).to_have_text(["run-1", "run-2", "run-10"])
    expect(page.locator('[data-sidebar="catalog"]')).to_contain_text("succeeded")
