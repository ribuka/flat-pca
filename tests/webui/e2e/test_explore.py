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


# Page y of the heatmap row at each StepTime of the fixture segment.
ROW_CENTERS_JS = """heatmap => {
    const box = heatmap.getBoundingClientRect();
    const layout = heatmap._fullLayout;
    return [0, 0.5, 1].map((value) => box.top + layout._size.t + layout.yaxis.l2p(value));
}"""


def test_vertical_step_time_slider_meets_the_rows(
    page: Page, cataloged_server_url: str
) -> None:
    """The StepTime slider stands beside the heatmap with its ends on the end rows."""
    _open(page, cataloged_server_url)
    slider = page.locator("#explore-step-time")
    expect(slider).to_have_css("writing-mode", "vertical-lr")

    rows = page.locator("#explore-heatmap").evaluate(ROW_CENTERS_JS)
    box = slider.bounding_box()
    assert box is not None
    thumb = box["width"]
    heatmap = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert heatmap is not None
    # Index 0 (StepTime 0) is at the bottom, the last index at the top.
    assert abs(box["y"] + box["height"] - thumb / 2 - rows[0]) < 1
    assert abs(box["y"] + thumb / 2 - rows[2]) < 1
    assert box["x"] + box["width"] <= heatmap["x"]


def _assert_slider_ends_on_rows(page: Page, first: int, last: int) -> None:
    """Assert that the StepTime slider spans indices first..last on their rows."""
    slider = page.locator("#explore-step-time")
    expect(slider).to_have_attribute("min", str(first))
    expect(slider).to_have_attribute("max", str(last))
    rows = page.locator("#explore-heatmap").evaluate(ROW_CENTERS_JS)
    box = slider.bounding_box()
    plot = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert box is not None
    assert plot is not None
    thumb = box["width"]
    assert abs(box["y"] + box["height"] - thumb / 2 - rows[first]) < 1
    assert abs(box["y"] + thumb / 2 - rows[last]) < 1
    assert plot["y"] <= box["y"]
    assert box["y"] + box["height"] <= plot["y"] + plot["height"]


def test_zoom_fits_the_step_time_slider_to_the_shown_rows(
    page: Page, cataloged_server_url: str
) -> None:
    """A drag zoom limits the slider to the shown StepTimes; a reset restores it."""
    _open(page, cataloged_server_url)
    _assert_slider_ends_on_rows(page, 0, 2)
    plot = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert plot is not None

    # Zoom to the upper half: StepTime 0.5 and 1 stay in view.
    page.mouse.move(plot["x"] + 5, plot["y"] + 5)
    page.mouse.down()
    page.mouse.move(plot["x"] + plot["width"] - 5, plot["y"] + plot["height"] * 0.55, steps=5)
    page.mouse.up()

    _assert_slider_ends_on_rows(page, 1, 2)
    page.locator("#explore-step-time").press("Home")
    expect(page.locator("#explore-step-time-value")).to_have_text("0.5")

    page.mouse.dblclick(plot["x"] + plot["width"] / 2, plot["y"] + plot["height"] / 2)

    _assert_slider_ends_on_rows(page, 0, 2)


def test_markers_point_at_the_selected_point(
    page: Page, cataloged_server_url: str
) -> None:
    """Triangles above and left of the heatmap follow the selected point."""
    _open(page, cataloged_server_url)
    annotations = "heatmap => heatmap.layout.annotations.map((a) => [a.text, a.x, a.y])"
    heatmap = page.locator("#explore-heatmap")
    expect(heatmap.locator(".annotation")).to_have_count(2)
    assert heatmap.evaluate(annotations) == [["▼", 401, 1], ["▶", 0, 0.5]]

    page.locator("#explore-step-time").press("End")

    expect(page.locator("#explore-step-time-value")).to_have_text("1")
    assert heatmap.evaluate(annotations) == [["▼", 401, 1], ["▶", 0, 1]]
