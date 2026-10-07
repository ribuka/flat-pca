"""Browser tests of the sidebar's collapse button, width choice, and memory."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

FIGURE = "#monitoring-t2"


def _open(page: Page, url: str) -> None:
    """Open a page with a figure and wait until it is drawn."""
    page.goto(url)
    expect(page.locator(FIGURE)).to_have_attribute("data-plot-ready", "true")


def _box(page: Page, selector: str) -> dict[str, float]:
    """Return the bounding box of the element matching ``selector``."""
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return box


def _figure_width(page: Page) -> float:
    """Return the width at which the figure is drawn."""
    return _box(page, f"{FIGURE} .main-svg >> nth=0")["width"]


def test_collapse_keeps_the_button_in_place_and_survives_navigation(
    page: Page, fitted_server_url: str
) -> None:
    """Collapsing hides the sidebar but not the button, which stays put."""
    page.set_viewport_size({"width": 1400, "height": 900})
    _open(page, f"{fitted_server_url}/monitoring")
    toggle = page.locator("#sidebar-toggle")
    expanded = _box(page, "#sidebar-toggle")
    width = _figure_width(page)

    toggle.click()

    expect(page.locator("#sidebar-body")).to_be_hidden()
    expect(page.locator("#sidebar-system")).to_be_hidden()
    expect(toggle).to_have_attribute("aria-expanded", "false")
    collapsed = _box(page, "#sidebar-toggle")
    assert (collapsed["x"], collapsed["y"]) == (expanded["x"], expanded["y"])
    page.wait_for_function(
        f"document.querySelector('{FIGURE} .main-svg').getBoundingClientRect().width > {width}"
    )

    page.goto(f"{fitted_server_url}/")
    expect(page.locator("#sidebar-body")).to_be_hidden()
    expect(toggle).to_have_attribute("aria-expanded", "false")

    toggle.click()

    expect(page.locator("#sidebar-body")).to_be_visible()
    reopened = _box(page, "#sidebar-toggle")
    assert (reopened["x"], reopened["y"]) == (expanded["x"], expanded["y"])


def test_width_choice_narrows_the_page_and_redraws_figures(
    page: Page, fitted_server_url: str
) -> None:
    """Compact width centers a narrower page; the figures follow its width."""
    page.set_viewport_size({"width": 1800, "height": 900})
    _open(page, f"{fitted_server_url}/monitoring")
    compact = page.locator('[data-width-choice="compact"]')
    wide = page.locator('[data-width-choice="wide"]')
    expect(wide).to_have_attribute("aria-pressed", "true")
    wide_content = _box(page, ".content")["width"]
    wide_figure = _figure_width(page)

    compact.click()

    expect(compact).to_have_attribute("aria-pressed", "true")
    expect(wide).to_have_attribute("aria-pressed", "false")
    assert _box(page, ".content")["width"] < wide_content
    page.wait_for_function(
        f"document.querySelector('{FIGURE} .main-svg').getBoundingClientRect().width < {wide_figure}"
    )

    _open(page, f"{fitted_server_url}/monitoring")
    expect(compact).to_have_attribute("aria-pressed", "true")
    assert _box(page, ".content")["width"] < wide_content

    wide.click()

    assert _box(page, ".content")["width"] == wide_content


def test_sidebar_shows_memory_and_version(page: Page, server_url: str) -> None:
    """The bottom of the sidebar shows the memory usage and the version."""
    page.goto(server_url)

    expect(page.locator('[data-sidebar="process-memory"]')).to_contain_text("GiB")
    expect(page.locator('[data-sidebar="system-memory"]')).to_contain_text("%")
    expect(page.locator('[data-sidebar="version"]')).to_have_text(
        re.compile(r"^v\d+\.\d+\.\d+")
    )


def test_page_restored_by_back_shows_the_choices_made_since(
    page: Page, fitted_server_url: str
) -> None:
    """A page restored from the bfcache takes the choices kept since it was left.

    Playwright's Chrome does not keep pages in the bfcache, so the test keeps
    other choices, as another page would, and dispatches the ``pageshow``
    event of a restored page.
    """
    _open(page, f"{fitted_server_url}/monitoring")
    width = _figure_width(page)

    page.evaluate(
        """() => {
          localStorage.setItem("flat-pca:sidebar", "collapsed");
          localStorage.setItem("flat-pca:width", "compact");
          window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
        }"""
    )

    root = page.locator("html")
    expect(root).to_have_attribute("data-sidebar", "collapsed")
    expect(root).to_have_attribute("data-width", "compact")
    expect(page.locator("#sidebar-body")).to_be_hidden()
    expect(page.locator("#sidebar-toggle")).to_have_attribute("aria-expanded", "false")
    expect(page.locator('[data-width-choice="compact"]')).to_have_attribute(
        "aria-pressed", "true"
    )
    page.wait_for_function(
        f"document.querySelector('{FIGURE} .main-svg').getBoundingClientRect().width"
        f" != {width}"
    )
