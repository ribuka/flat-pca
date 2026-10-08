"""Browser tests of the T² and Q page."""

from __future__ import annotations

import math
import re

import pytest
from playwright.sync_api import FloatRect, Page, expect

pytestmark = pytest.mark.e2e

# Longer than Plotly's double-click delay (300 ms).
DOUBLE_CLICK_DELAY_MS = 400
FIGURES = ("#monitoring-t2", "#monitoring-q", "#monitoring-scatter")


def _open(page: Page, url: str) -> None:
    """Open a T² and Q page and wait until every figure is drawn."""
    page.goto(url)
    for target in FIGURES:
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")


def _plot_area(page: Page, target: str) -> FloatRect:
    """Return the box of a figure's plot area."""
    box = page.locator(f"{target} .nsewdrag").first.bounding_box()
    assert box is not None
    return box


def _empty_spot(page: Page, target: str) -> tuple[float, float]:
    """Return a position in a figure's plot area away from every point.

    Plotly reports a click on the nearest point within its hover distance,
    so a double click near a point would also click that point.
    """
    area = _plot_area(page, target)
    centers = []
    for point in page.locator(f"{target} .scatterlayer .point").all():
        box = point.bounding_box()
        assert box is not None
        centers.append((box["x"] + box["width"] / 2, box["y"] + box["height"] / 2))
    spots = [
        (area["x"] + area["width"] * (i + 0.5) / 10, area["y"] + area["height"] * (j + 0.5) / 10)
        for i in range(10)
        for j in range(10)
    ]
    return max(
        spots,
        key=lambda spot: min(math.dist(spot, center) for center in centers),
    )


def test_clicking_a_point_shows_its_row_and_stays(page: Page, fitted_server_url: str) -> None:
    """A click on a point of each figure shows that file's row and keeps the page."""
    url = f"{fitted_server_url}/monitoring"
    _open(page, url)
    table = page.locator("#point-table")
    expect(table.locator("tbody tr")).to_have_count(0)
    expect(table.locator("[data-point-table-empty]")).to_be_visible()

    for target in FIGURES:
        page.locator(target).scroll_into_view_if_needed()
        # The plot's drag layer covers the points, so click at the point's position.
        box = page.locator(f"{target} .scatterlayer .point").first.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

        rows = table.locator("tbody tr")
        expect(rows).to_have_count(1)
        expect(rows.first.locator("td").first).to_have_text(re.compile(r"s-\d{2}"))
        expect(table.locator("[data-point-table-empty]")).to_be_hidden()

        # A double click in the default zoom mode clears the table. Plotly
        # would join a click within its double-click delay to the last click.
        page.wait_for_timeout(DOUBLE_CLICK_DELAY_MS)
        page.mouse.dblclick(*_empty_spot(page, target))

        expect(table).to_have_attribute("data-shown-count", "0")
        expect(table.locator("[data-point-table-empty]")).to_be_visible()
    expect(table.locator("th")).to_have_text(
        ["file", "lot", "date", "yield_pct", "T²", "Q", "T² above UCL", "Q above UCL"]
    )
    assert page.url == url
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(0)


def test_box_select_shows_every_chosen_row(page: Page, fitted_server_url: str) -> None:
    """A box over the whole chart shows every file; a double click clears the table."""
    _open(page, f"{fitted_server_url}/monitoring")
    table = page.locator("#point-table")
    expect(page.locator('#monitoring-scatter [data-title="Lasso Select"]')).to_have_count(1)

    page.locator("#monitoring-t2 .nsewdrag").first.hover()
    page.locator('#monitoring-t2 [data-title="Box Select"]').click()
    area = _plot_area(page, "#monitoring-t2")
    page.mouse.move(area["x"] + 2, area["y"] + 2)
    page.mouse.down()
    page.mouse.move(area["x"] + area["width"] / 2, area["y"] + area["height"] / 2, steps=5)
    page.mouse.move(area["x"] + area["width"] - 2, area["y"] + area["height"] - 2, steps=5)
    page.mouse.up()

    expect(table).to_have_attribute("data-shown-count", "12")
    expect(table.locator("tbody tr")).to_have_count(12)

    page.mouse.dblclick(area["x"] + area["width"] / 2, area["y"] + area["height"] / 2)

    expect(table).to_have_attribute("data-shown-count", "0")
    expect(table.locator("[data-point-table-empty]")).to_be_visible()


def test_choosing_the_order_reloads_the_charts(page: Page, fitted_server_url: str) -> None:
    """Choosing an ordering column reloads the page with it."""
    _open(page, f"{fitted_server_url}/monitoring")

    page.locator('select[name="order"]').select_option("lot")

    expect(page).to_have_url(f"{fitted_server_url}/monitoring?color=lot&order=lot")
    _open(page, page.url)
    expect(page.locator("#monitoring-q .xtitle")).to_have_text("file order (lot)")


def test_choosing_the_color_recolors_every_figure(page: Page, fitted_server_url: str) -> None:
    """Choosing a coloring column reloads the page with every figure colored by it."""
    _open(page, f"{fitted_server_url}/monitoring?color=")

    page.locator('select[name="color"]').select_option("lot")

    expect(page).to_have_url(f"{fitted_server_url}/monitoring?color=lot&order=date")
    _open(page, page.url)
    for name in ("t2", "q", "scatter"):
        title = page.locator(f"#monitoring-{name}").evaluate("plot => plot.layout.legend.title.text")
        assert title == "lot"
