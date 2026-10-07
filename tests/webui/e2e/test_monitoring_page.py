"""Browser tests of the T² and Q page."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

FIGURES = ("#monitoring-t2", "#monitoring-q", "#monitoring-scatter")


def _open(page: Page, url: str) -> None:
    """Open a T² and Q page and wait until every figure is drawn."""
    page.goto(url)
    for target in FIGURES:
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")


def test_clicking_a_control_chart_point_opens_the_q_contribution(
    page: Page, fitted_server_url: str
) -> None:
    """A click on a Q chart point opens that file's Q contribution heatmap."""
    _open(page, f"{fitted_server_url}/monitoring")

    # The plot's drag layer covers the points, so click at the point's position.
    box = page.locator("#monitoring-q .scatterlayer .point").first.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    expect(page).to_have_url(
        re.compile(
            re.escape(f"{fitted_server_url}/explore?view=q_contribution&run=fit-1&file=")
            + r"s-\d{2}$"
        )
    )
    stem = page.url.rsplit("=", 1)[1]
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", stem)
    expect(page.locator("[data-view-note]")).to_contain_text("先頭 1..2 成分")


def test_choosing_the_order_reloads_the_charts(page: Page, fitted_server_url: str) -> None:
    """Choosing an ordering column reloads the page with it."""
    _open(page, f"{fitted_server_url}/monitoring")

    page.locator('select[name="order"]').select_option("lot")

    expect(page).to_have_url(f"{fitted_server_url}/monitoring?run=fit-1&order=lot")
    _open(page, page.url)
    expect(page.locator("#monitoring-q .xtitle")).to_have_text("file order (lot)")
