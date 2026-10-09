"""Browser tests of the score page."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import (
    choose_files,
    mark_page,
    open_file_dialog,
    select_in_dialog,
    shown_files,
    wait_for_sidebar,
    wait_for_view_refresh,
)
from spectra import SPECTRA_SHORT_FILE

pytestmark = pytest.mark.e2e

SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"


def _open(page: Page, url: str) -> None:
    """Open a score page and wait until the score scatter plot is drawn."""
    page.goto(url)
    expect(page.locator("#scores-scatter")).to_have_attribute("data-plot-ready", "true")
    wait_for_sidebar(page)


def _click_point(page: Page, index: int) -> None:
    """Click one point of the score scatter plot."""
    # The plot's drag layer covers the points, so click at the point's position.
    box = page.locator("#scores-scatter .scatterlayer .point").nth(index).bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_clicking_a_score_point_adds_the_file(page: Page, fitted_server_url: str) -> None:
    """A click on a score point adds the file to the sidebar without a reload."""
    url = f"{fitted_server_url}/scores?color="
    _open(page, url)
    expect(shown_files(page)).to_have_count(0)

    mark_page(page)
    _click_point(page, 1)

    wait_for_view_refresh(page)
    expect(page).to_have_url(url)
    # The replaced point table shows the row of the clicked file.
    rows = page.locator("#point-table tbody tr")
    expect(rows).to_have_count(1)
    expect(rows.first.locator("td").first).to_have_text("s-01")
    expect(shown_files(page)).to_have_text(["s-01"])
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(1)
    expect(legend.nth(0)).to_contain_text("s-01")

    # The replaced figure is drawn, and its double click clears the row.
    expect(page.locator("#scores-scatter")).to_have_attribute("data-plot-ready", "true")
    box = page.locator("#scores-scatter .nsewdrag").first.bounding_box()
    assert box is not None
    page.mouse.dblclick(box["x"] + 4, box["y"] + 4)
    expect(page.locator("#point-table")).to_have_attribute("data-shown-count", "0")
    # The dialog starts from the added file.
    dialog = open_file_dialog(page)
    expect(dialog.locator("[data-dt-row-check]:checked")).to_have_count(1)
    expect(dialog.get_by_label("Select s-01", exact=True)).to_be_checked()


def test_clicking_a_point_at_the_limit_warns(page: Page, one_file_server_url: str) -> None:
    """At the limit of shown files, a clicked file is not added and a warning is shown."""
    choose_files(page, one_file_server_url, ["s-00"])
    url = f"{one_file_server_url}/scores?color="
    _open(page, url)
    expect(shown_files(page)).to_have_text(["s-00"])

    _click_point(page, 1)

    warning = page.locator("#plot-select-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Up to 1 shown files")
    expect(warning).to_contain_text("s-01")
    assert page.url == url
    expect(shown_files(page)).to_have_count(1)
    rows = page.locator("#point-table tbody tr")
    expect(rows).to_have_count(1)
    expect(rows.first.locator("td").first).to_have_text("s-01")
    # The row is not kept for a later load of the page.
    _open(page, url)
    expect(page.locator("#point-table tbody tr")).to_have_count(0)


def test_lasso_select_shows_the_rows_without_adding(
    page: Page, fitted_server_url: str
) -> None:
    """A lasso around every point shows each file with its scores and adds nothing."""
    url = f"{fitted_server_url}/scores?color="
    _open(page, url)

    page.locator("#scores-scatter .nsewdrag").first.hover()
    page.locator('#scores-scatter [data-title="Lasso Select"]').click()
    box = page.locator("#scores-scatter .nsewdrag").first.bounding_box()
    assert box is not None
    left, top = box["x"] + 2, box["y"] + 2
    right, bottom = box["x"] + box["width"] - 2, box["y"] + box["height"] - 2
    page.mouse.move(left, top)
    page.mouse.down()
    for x, y in ((right, top), (right, bottom), (left, bottom), (left, top + 4)):
        page.mouse.move(x, y, steps=5)
    page.mouse.up()

    table = page.locator("#point-table")
    expect(table).to_have_attribute("data-shown-count", "12")
    expect(table.locator("th")).to_have_text(
        ["file", "lot", "date", "yield_pct", "PC1", "PC2"]
    )
    assert page.url == url
    expect(shown_files(page)).to_have_count(0)


def test_choices_redraw_the_trajectories(page: Page, fitted_server_url: str) -> None:
    """Files chosen in the sidebar and another component redraw the trajectories."""
    _open(page, f"{fitted_server_url}/scores")
    expect(page.locator("#scores-trajectories .legend .traces")).to_have_count(1)
    expect(page.locator('select[name="file"]')).to_have_count(0)
    expect(page.locator('select[name="run"]')).to_have_count(0)

    mark_page(page)
    for stem in ("s-00", SHORT):
        select_in_dialog(page, [stem])
        wait_for_view_refresh(page)
    expect(page.locator("#scores-scatter")).to_have_attribute("data-plot-ready", "true")
    y_input = page.locator('input[name="trajectory_y"]')
    y_input.fill("3")
    y_input.dispatch_event("change")

    expect(page).to_have_url(
        f"{fitted_server_url}/scores?x=1&y=2&color=lot"
        "&trajectory_x=1&trajectory_y=3&trajectory_color=stem"
    )
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(2)
    expect(legend.nth(1)).to_contain_text(SHORT)
    expect(page.locator("#scores-trajectories .xtitle")).to_have_text("PC1")
    expect(page.locator("#scores-trajectories .ytitle")).to_have_text("PC3")
    # The score scatter plot keeps its own components.
    expect(page.locator("#scores-scatter .ytitle")).to_have_text("PC2")


def test_trajectory_color_groups_the_files(page: Page, fitted_server_url: str) -> None:
    """Coloring the trajectories by a column shows one legend entry per value.

    The files of this workspace have no lot, so all three share the missing value.
    """
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    _open(page, f"{fitted_server_url}/scores")
    expect(page.locator("#scores-trajectories .legend .traces")).to_have_count(3)

    page.locator('select[name="trajectory_color"]').select_option("lot")

    expect(page).to_have_url(re.compile(r"trajectory_color=lot"))
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(1)
    expect(legend.first).to_contain_text("(missing)")
    expect(page.locator("#scores-trajectories .legendtitletext")).to_have_text("lot")


def test_a_failed_click_request_releases_the_page(
    page: Page, fitted_server_url: str
) -> None:
    """When the request of a clicked point fails, the overlay goes and a warning shows."""
    url = f"{fitted_server_url}/scores?color="
    _open(page, url)
    page.evaluate("() => { window.fetch = () => Promise.reject(new TypeError('offline')); }")

    _click_point(page, 1)

    warning = page.locator("#plot-select-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Could not add s-01 to the shown files")
    expect(page.locator("#busy-overlay")).to_be_hidden()
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")
    assert page.url == url


def _frame_size(page: Page) -> tuple[float, float]:
    """Return the width and height of the score scatter plot's frame."""
    box = page.locator("#scores-scatter .nsewdrag").first.bounding_box()
    assert box is not None
    return box["width"], box["height"]


def test_score_frame_stays_square(page: Page, fitted_server_url: str) -> None:
    """The frame stays square after Autoscale and at another window width."""
    _open(page, f"{fitted_server_url}/scores")
    width, height = _frame_size(page)
    assert width == pytest.approx(height, abs=1)

    page.locator("#scores-scatter .nsewdrag").first.hover()
    page.locator('#scores-scatter [data-title="Autoscale"]').click()
    page.set_viewport_size({"width": 700, "height": 800})

    assert _frame_size(page) == pytest.approx((width, height), abs=1)
