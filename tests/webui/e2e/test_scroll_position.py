"""Browser tests that the page keeps its scroll position after an operation."""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

pytestmark = pytest.mark.e2e

# Short enough that every tested page scrolls.
VIEWPORT = {"width": 1200, "height": 500}


def _scroll_to(control: Locator) -> float:
    """Scroll the page so that a control is at the top of the viewport.

    Returns
    -------
    float
        The page's scroll position, which must be away from the top.
    """
    control.evaluate("(element) => element.scrollIntoView({ block: 'start' })")
    y = control.page.evaluate("() => window.scrollY")
    assert y > 100
    return y


def _expect_scroll(page: Page, y: float) -> None:
    """Wait until the page is scrolled to ``y`` (within a pixel)."""
    page.wait_for_function("(y) => Math.abs(window.scrollY - y) < 1", arg=y)


@pytest.mark.parametrize(
    ("path", "select", "value", "figure"),
    [
        ("/scores", 'select[name="trajectory_color"]', "lot", "#scores-trajectories"),
        ("/model", 'select[name="color_by"]', "StepTime", "#model-loadings"),
    ],
)
def test_choice_that_loads_the_page_keeps_the_scroll_position(
    page: Page, fitted_server_url: str, path: str, select: str, value: str, figure: str
) -> None:
    """A choice of a display screen loads its next page at the same scroll position."""
    page.set_viewport_size(VIEWPORT)
    page.goto(f"{fitted_server_url}{path}")
    expect(page.locator(figure)).to_have_attribute("data-plot-ready", "true")
    control = page.locator(select)
    y = _scroll_to(control)

    with page.expect_navigation():
        control.select_option(value)

    expect(page.locator(select)).to_have_value(value)
    expect(page.locator(figure)).to_have_attribute("data-plot-ready", "true")
    _expect_scroll(page, y)


def test_screen_opened_from_the_sidebar_starts_at_the_top(
    page: Page, fitted_server_url: str
) -> None:
    """Only the page loaded by a choice scrolls back; opening a screen starts at the top."""
    page.set_viewport_size(VIEWPORT)
    page.goto(f"{fitted_server_url}/scores")
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    control = page.locator('select[name="trajectory_color"]')
    _scroll_to(control)
    with page.expect_navigation():
        control.select_option("lot")
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")

    with page.expect_navigation():
        page.locator(".sidebar-nav").get_by_role("link", name="Scores").click()

    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    assert page.evaluate("() => window.scrollY") == 0


def test_fit_form_change_keeps_the_scroll_position(
    page: Page, cataloged_server_url: str
) -> None:
    """A change of the fit form, which updates the estimate, keeps the scroll position."""
    page.set_viewport_size(VIEWPORT)
    page.goto(f"{cataloged_server_url}/fit")
    checkbox = page.locator("#fit-form input[type=checkbox]:enabled").last
    y = _scroll_to(checkbox)

    with page.expect_response("**/fit/estimate"):
        checkbox.click()

    page.wait_for_timeout(500)
    _expect_scroll(page, y)


def test_selecting_files_keeps_the_scroll_position(
    page: Page, cataloged_server_url: str
) -> None:
    """"Select" on the data selection page keeps the scroll position."""
    page.set_viewport_size(VIEWPORT)
    page.goto(cataloged_server_url)
    expect(page.locator("#files tbody tr")).to_have_count(3)
    page.locator("#files tbody tr").first.click()
    button = page.locator("#select-files")
    y = _scroll_to(button)

    button.click()

    expect(page.locator("#selection-summary")).to_contain_text("Fit target: 1 files")
    _expect_scroll(page, y)
