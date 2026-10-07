"""Browser tests of the spectral exploration page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

# Segment (1, 1) of the fixture files: StepTime 0, 0.5, 1 over 400, 401, 402.5 nm.
EXPLORE_PATH = "/explore?view=raw&file=run-1&file=run-2&segment=1:1"


def _open(page: Page, base_url: str) -> None:
    """Open the raw view and wait for the initial trends at the middle point."""
    page.goto(base_url + EXPLORE_PATH)
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "401")
    expect(root).to_have_attribute("data-trend-step-time", "0.5")


def test_heatmap_click_moves_sliders_crosshair_and_trends(
    page: Page, cataloged_server_url: str
) -> None:
    """Clicking a heatmap cell selects its point everywhere."""
    _open(page, cataloged_server_url)
    expect(page.locator("#explore-trend-step-time .scatterlayer .trace")).to_have_count(2)
    expect(page.locator("#explore-heatmap .shapelayer path")).to_have_count(2)

    plot_area = page.locator("#explore-heatmap .nsewdrag")
    box = plot_area.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.9)

    expect(page.locator("#explore-wavelength")).to_have_value("0")
    expect(page.locator("#explore-step-time")).to_have_value("0")
    expect(page.locator("#explore-wavelength-value")).to_have_text("400")
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "400")
    expect(root).to_have_attribute("data-trend-step-time", "0")


def test_sliders_move_the_point_and_reload_the_trends(
    page: Page, cataloged_server_url: str
) -> None:
    """The wavelength and StepTime sliders choose the point of the trends."""
    _open(page, cataloged_server_url)

    page.locator("#explore-wavelength").press("ArrowRight")
    page.locator("#explore-step-time").press("End")

    expect(page.locator("#explore-wavelength-value")).to_have_text("402.5")
    expect(page.locator("#explore-step-time-value")).to_have_text("1")
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "402.5")
    expect(root).to_have_attribute("data-trend-step-time", "1")
    expect(page.locator("#explore-trend-wavelength .gtitle")).to_have_text("StepTime = 1")
