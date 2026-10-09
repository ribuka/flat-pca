"""Browser tests of the spectral exploration page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import choose_files

pytestmark = pytest.mark.e2e

# Segment (1, 1) of the synthetic spectra: StepTime 0..3 over 400, 401, 402.5, 405, 410 nm.
EXPLORE_PATH = "/explore?view=raw&segment=1:1"


def _open(page: Page, base_url: str) -> None:
    """Open the raw view of two files and wait for the initial trends."""
    choose_files(page, base_url, ["s-00", "s-01"])
    page.goto(base_url + EXPLORE_PATH)
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "402.5")
    expect(root).to_have_attribute("data-trend-step-time", "1")


def test_heatmap_click_moves_sliders_crosshair_and_trends(
    page: Page, fitted_server_url: str
) -> None:
    """Clicking a heatmap cell selects its point everywhere."""
    _open(page, fitted_server_url)
    expect(page.locator("#explore-trend-step-time .scatterlayer .trace")).to_have_count(2)
    expect(page.locator("#explore-heatmap .shapelayer path")).to_have_count(2)

    plot_area = page.locator("#explore-heatmap .nsewdrag")
    box = plot_area.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] * 0.03, box["y"] + box["height"] * 0.9)

    expect(page.locator("#explore-wavelength")).to_have_value("400")
    expect(page.locator("#explore-step-time")).to_have_value("0")
    expect(page.locator("#explore-wavelength-value")).to_have_text("400")
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "400")
    expect(root).to_have_attribute("data-trend-step-time", "0")


def test_sliders_move_the_point_and_reload_the_trends(
    page: Page, fitted_server_url: str
) -> None:
    """The wavelength and StepTime sliders choose the point of the trends."""
    _open(page, fitted_server_url)

    page.locator("#explore-wavelength").press("ArrowRight")
    page.locator("#explore-step-time").press("End")

    expect(page.locator("#explore-wavelength-value")).to_have_text("405")
    expect(page.locator("#explore-step-time")).to_have_value("3")
    # Only the wavelength shows its value; the StepTime value is not shown.
    expect(page.locator("[data-explore] output")).to_have_count(1)
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "405")
    expect(root).to_have_attribute("data-trend-step-time", "3")
    expect(page.locator("#explore-trend-wavelength .gtitle")).to_have_text("StepTime = 3")


# Page y of the heatmap row at each StepTime of the fixture segment.
ROW_CENTERS_JS = """heatmap => {
    const box = heatmap.getBoundingClientRect();
    const layout = heatmap._fullLayout;
    return [0, 1, 2, 3].map((value) => box.top + layout._size.t + layout.yaxis.l2p(value));
}"""


def test_vertical_step_time_slider_meets_the_rows(
    page: Page, fitted_server_url: str
) -> None:
    """The StepTime slider stands beside the heatmap with its ends on the end rows."""
    _open(page, fitted_server_url)
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
    assert abs(box["y"] + thumb / 2 - rows[3]) < 1
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
    page: Page, fitted_server_url: str
) -> None:
    """A drag zoom limits the slider to the shown StepTimes; a reset restores it."""
    _open(page, fitted_server_url)
    _assert_slider_ends_on_rows(page, 0, 3)
    plot = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert plot is not None

    # Zoom to the upper half: StepTime 2 and 3 stay in view.
    page.mouse.move(plot["x"] + 5, plot["y"] + 5)
    page.mouse.down()
    page.mouse.move(plot["x"] + plot["width"] - 5, plot["y"] + plot["height"] * 0.55, steps=5)
    page.mouse.up()

    _assert_slider_ends_on_rows(page, 2, 3)
    page.locator("#explore-step-time").press("Home")
    expect(page.locator("#explore-step-time")).to_have_value("2")
    expect(page.locator("[data-explore]")).to_have_attribute("data-trend-step-time", "2")

    page.mouse.dblclick(plot["x"] + plot["width"] / 2, plot["y"] + plot["height"] / 2)

    _assert_slider_ends_on_rows(page, 0, 3)


# The unevenly spaced wavelengths of the fixture segment, as slider values.
WAVELENGTHS = ["400", "401", "402.5", "405", "410"]

# Page x of the heatmap column at each wavelength of the fixture segment.
COLUMN_CENTERS_JS = """heatmap => {
    const box = heatmap.getBoundingClientRect();
    const layout = heatmap._fullLayout;
    return [400, 401, 402.5, 405, 410].map((value) => box.left + layout._size.l + layout.xaxis.l2p(value));
}"""


def _assert_wavelength_slider_ends_on_columns(page: Page, first: int, last: int) -> None:
    """Assert that the wavelength slider spans indices first..last on their columns."""
    slider = page.locator("#explore-wavelength")
    expect(slider).to_have_attribute("min", WAVELENGTHS[first])
    expect(slider).to_have_attribute("max", WAVELENGTHS[last])
    columns = page.locator("#explore-heatmap").evaluate(COLUMN_CENTERS_JS)
    box = slider.bounding_box()
    plot = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert box is not None
    assert plot is not None
    thumb = box["height"]
    assert abs(box["x"] + thumb / 2 - columns[first]) < 1
    assert abs(box["x"] + box["width"] - thumb / 2 - columns[last]) < 1
    assert plot["x"] <= box["x"]
    assert box["x"] + box["width"] <= plot["x"] + plot["width"]
    assert box["y"] >= plot["y"] + plot["height"]


def test_wavelength_slider_meets_the_columns_after_resize_and_zoom(
    page: Page, fitted_server_url: str
) -> None:
    """The wavelength slider below the heatmap keeps its ends on the shown end columns."""
    _open(page, fitted_server_url)
    _assert_wavelength_slider_ends_on_columns(page, 0, 4)

    page.set_viewport_size({"width": 900, "height": 900})
    # The responsive heatmap redraws to the new width after a debounce.
    page.wait_for_function(
        """() => {
            const heatmap = document.querySelector("#explore-heatmap");
            return Math.abs(heatmap._fullLayout.width - heatmap.clientWidth) < 1;
        }"""
    )
    _assert_wavelength_slider_ends_on_columns(page, 0, 4)

    # Zoom to the columns right of the middle of 402.5 nm and 405 nm.
    columns = page.locator("#explore-heatmap").evaluate(COLUMN_CENTERS_JS)
    plot = page.locator("#explore-heatmap .nsewdrag").bounding_box()
    assert plot is not None
    page.mouse.move((columns[2] + columns[3]) / 2, plot["y"] + 5)
    page.mouse.down()
    page.mouse.move(plot["x"] + plot["width"] - 5, plot["y"] + plot["height"] - 5, steps=5)
    page.mouse.up()

    _assert_wavelength_slider_ends_on_columns(page, 3, 4)
    # The selected 402.5 nm is hidden, so the point moves into the shown range.
    expect(page.locator("#explore-wavelength")).to_have_value("405")
    expect(page.locator("[data-explore]")).to_have_attribute("data-trend-wavelength", "405")

    page.mouse.dblclick(plot["x"] + plot["width"] / 2, plot["y"] + plot["height"] / 2)

    _assert_wavelength_slider_ends_on_columns(page, 0, 4)


def test_wavelength_slider_thumb_lies_on_the_selected_column(
    page: Page, fitted_server_url: str
) -> None:
    """With unevenly spaced wavelengths, the thumb and a click follow the columns."""
    _open(page, fitted_server_url)
    slider = page.locator("#explore-wavelength")
    slider.scroll_into_view_if_needed()
    columns = page.locator("#explore-heatmap").evaluate(COLUMN_CENTERS_JS)
    box = slider.bounding_box()
    assert box is not None
    thumb = box["height"]
    middle = box["y"] + box["height"] / 2

    # The thumb of the native slider sits linearly between its ends.
    def thumb_center() -> float:
        value, low, high = slider.evaluate("s => [s.value, s.min, s.max].map(Number)")
        return box["x"] + thumb / 2 + (box["width"] - thumb) * (value - low) / (high - low)

    expect(slider).to_have_value("402.5")
    assert abs(thumb_center() - columns[2]) < 1

    slider.press("ArrowRight")
    expect(slider).to_have_value("405")
    assert abs(thumb_center() - columns[3]) < 1

    # A click below a column selects that column's wavelength.
    page.mouse.click(columns[1] + 2, middle)
    expect(slider).to_have_value("401")
    expect(page.locator("[data-explore]")).to_have_attribute("data-trend-wavelength", "401")
    page.mouse.click(columns[2] + 2, middle)
    expect(slider).to_have_value("402.5")
    expect(page.locator("[data-explore]")).to_have_attribute("data-trend-wavelength", "402.5")


def test_markers_point_at_the_selected_point(
    page: Page, fitted_server_url: str
) -> None:
    """Triangles above and left of the heatmap follow the selected point."""
    _open(page, fitted_server_url)
    annotations = "heatmap => heatmap.layout.annotations.map((a) => [a.text, a.x, a.y])"
    heatmap = page.locator("#explore-heatmap")
    expect(heatmap.locator(".annotation")).to_have_count(2)
    assert heatmap.evaluate(annotations) == [["▼", 402.5, 1], ["▶", 0, 1]]

    page.locator("#explore-step-time").press("End")

    expect(page.locator("[data-explore]")).to_have_attribute("data-trend-step-time", "3")
    assert heatmap.evaluate(annotations) == [["▼", 402.5, 1], ["▶", 0, 3]]


def test_heatmap_file_choice_redraws_and_survives_a_reload(
    page: Page, fitted_server_url: str
) -> None:
    """The heatmap file pull-down redraws the heatmap, untitled, and stays in the URL."""
    _open(page, fitted_server_url)
    heatmap = page.locator("#explore-heatmap")
    expect(heatmap).to_have_attribute("data-heatmap-label", "s-00")
    expect(heatmap.locator(".gtitle")).to_have_count(0)

    page.locator('select[name="heatmap_file"]').select_option("s-01")

    expect(heatmap).to_have_attribute("data-heatmap-label", "s-01")
    assert "heatmap_file=s-01" in page.url
    expect(page.locator("#explore-trend-step-time .scatterlayer .trace")).to_have_count(2)
    page.reload()
    expect(heatmap).to_have_attribute("data-heatmap-label", "s-01")
    expect(page.locator('select[name="heatmap_file"]')).to_have_value("s-01")
    # The marker above the plot area stays inside the heatmap's top margin.
    marker = heatmap.locator(".annotation").first.bounding_box()
    box = heatmap.bounding_box()
    assert marker is not None
    assert box is not None
    assert marker["y"] >= box["y"]


# The trend figure height without legend, and its plot area height
# (TREND_FRAME_HEIGHT and trend_height of flat_pca.visualize.trend).
TREND_FRAME_HEIGHT = 320
TREND_BOXES_JS = """(target) => {
    const gd = document.querySelector(target);
    const box = (node) => {
        const rect = node.getBoundingClientRect();
        return { top: rect.top, bottom: rect.bottom };
    };
    return {
        frame: gd._fullLayout._size.h,
        rows: new Set([...gd.querySelectorAll(".legend .traces")].map((node) => box(node).top)).size,
        title: box(gd.querySelector(".gtitle")),
        legend: box(gd.querySelector(".legend")),
        plot: box(gd.querySelector(".nsewdrag")),
        div: box(gd),
    };
}"""


def _trend_boxes(page: Page, target: str) -> dict:
    """Return the plot area height, legend rows, and boxes of a trend figure."""
    return page.evaluate(TREND_BOXES_JS, target)


def _assert_legend_above_a_fixed_frame(boxes: dict) -> None:
    """Assert the title, legend, and plot area lie in order with a fixed plot area."""
    assert abs(boxes["frame"] - TREND_FRAME_HEIGHT) < 1
    assert boxes["title"]["bottom"] <= boxes["legend"]["top"]
    assert boxes["legend"]["bottom"] <= boxes["plot"]["top"] + 1


def test_trends_are_stacked_with_the_legend_above_a_fixed_plot_area(
    page: Page, fitted_server_url: str
) -> None:
    """The trends lie one under the other; a legend of more rows adds height, not plot area."""
    page.set_viewport_size({"width": 1400, "height": 900})
    _open(page, fitted_server_url)
    step_time = _trend_boxes(page, "#explore-trend-step-time")
    wavelength = _trend_boxes(page, "#explore-trend-wavelength")
    assert step_time["div"]["bottom"] <= wavelength["div"]["top"]
    assert step_time["rows"] == 1
    _assert_legend_above_a_fixed_frame(step_time)
    _assert_legend_above_a_fixed_frame(wavelength)

    page.set_viewport_size({"width": 700, "height": 900})
    choose_files(page, fitted_server_url, [f"s-{index:02d}" for index in range(12)])
    page.goto(fitted_server_url + EXPLORE_PATH)
    expect(page.locator("#explore-trend-wavelength .scatterlayer .trace")).to_have_count(12)
    page.wait_for_function(
        "() => document.querySelector('#explore-trend-wavelength')._fullLayout._size.h > 319"
    )
    many = _trend_boxes(page, "#explore-trend-step-time")
    assert many["rows"] > 1
    _assert_legend_above_a_fixed_frame(many)
    _assert_legend_above_a_fixed_frame(_trend_boxes(page, "#explore-trend-wavelength"))

    # A wider page wraps the legend into fewer rows; the plot area keeps its height.
    page.set_viewport_size({"width": 1400, "height": 900})
    page.wait_for_function(
        """(rows) => {
            const gd = document.querySelector("#explore-trend-step-time");
            const shown = new Set([...gd.querySelectorAll(".legend .traces")]
                .map((node) => node.getBoundingClientRect().top)).size;
            return shown < rows && Math.abs(gd._fullLayout.width - gd.clientWidth) < 1;
        }""",
        arg=many["rows"],
    )
    page.wait_for_function(
        "() => Math.abs(document.querySelector('#explore-trend-step-time')._fullLayout._size.h - 320) < 1"
    )
    _assert_legend_above_a_fixed_frame(_trend_boxes(page, "#explore-trend-step-time"))


def test_legend_room_layout_adds_the_legend_to_height_and_top_margin(
    page: Page, fitted_server_url: str
) -> None:
    """The legend room rounds the legend up and adds it to the height and the top margin."""
    page.goto(fitted_server_url + EXPLORE_PATH)

    room = page.evaluate("() => legendRoomLayout({ height: 416, margin: { t: 40 } }, 28.2)")

    assert room == {"height": 445, "margin.t": 69, "autosize": True}


# Draws a trend of the page's layout with `count` long-labeled traces in a new
# figure and fits it as the page does; returns the legend and plot area sizes.
TALL_LEGEND_JS = """async (count) => {
    const root = document.querySelector("[data-explore]");
    const url = `${root.dataset.trendUrl}&wavelength=400&step_time=0`;
    const layout = (await (await fetch(url)).json()).by_step_time.layout;
    const base = structuredClone(layout);
    const data = Array.from({ length: count }, (_, index) => ({
        x: [0, 1], y: [index, index + 1], type: "scatter",
        name: `measurement-${String(index).padStart(2, "0")}-with-a-rather-long-label`,
    }));
    const gd = document.createElement("div");
    gd.style.width = "500px";
    document.body.append(gd);
    await Plotly.react(gd, data, layout);
    await fitLegendRoom(gd, base);
    return {
        base: base.height,
        legend: gd._fullLayout.legend._height,
        shown: gd.querySelector(".legend .bg").getBBox().height,
        frame: gd._fullLayout._size.h,
    };
}"""


def test_a_legend_taller_than_the_figure_is_shown_whole(
    page: Page, fitted_server_url: str
) -> None:
    """A legend taller than the figure without it is shown whole above a fixed plot area."""
    _open(page, fitted_server_url)

    sizes = page.evaluate(TALL_LEGEND_JS, 40)

    assert sizes["legend"] > sizes["base"]
    assert abs(sizes["shown"] - sizes["legend"]) < 1
    assert abs(sizes["frame"] - TREND_FRAME_HEIGHT) < 1
