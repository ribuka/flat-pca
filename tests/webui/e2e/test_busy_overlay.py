"""Browser tests of the overlay shown while an auto-submitted page loads."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

FIGURES = ("#monitoring-t2", "#monitoring-q", "#monitoring-scatter")


def _open(page: Page, url: str) -> None:
    """Open a T² and Q page and wait until every figure is drawn."""
    page.goto(url)
    for target in FIGURES:
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")


def _keep_submissions_pending(page: Page) -> None:
    """Cancel the navigation of every form submission on the page.

    The submit handler of the page still runs, so the page stays as it is
    while the next page would load. Playwright cannot inspect a page whose
    navigation is pending, so the tests use this in place of a slow server.
    """
    page.evaluate(
        """() => {
          for (const form of document.querySelectorAll("form")) {
            form.addEventListener("submit", (event) => event.preventDefault());
          }
        }"""
    )


def test_pending_page_shows_overlay_and_cancel_restores_the_conditions(
    page: Page, fitted_server_url: str
) -> None:
    """A pending load greys out the page; cancelling keeps the shown page."""
    url = f"{fitted_server_url}/monitoring"
    _open(page, url)
    order = page.locator('select[name="order"]')
    shown_order = order.input_value()
    _keep_submissions_pending(page)

    order.select_option("lot")

    overlay = page.locator("#busy-overlay")
    layout = page.locator(".layout")
    expect(overlay).to_have_class("busy-overlay busy-visible")
    expect(page.locator(".busy-spinner")).to_be_visible()
    expect(page.locator("#busy-elapsed")).to_have_text("1")
    expect(layout).to_have_attribute("inert", "")

    page.locator("#busy-cancel").click()

    expect(overlay).to_be_hidden()
    expect(order).to_have_value(shown_order)
    expect(layout).not_to_have_attribute("inert", "")
    assert page.url == url


def test_going_back_leaves_no_overlay(page: Page, fitted_server_url: str) -> None:
    """The page restored by "back" shows its own conditions without the overlay."""
    url = f"{fitted_server_url}/monitoring"
    _open(page, url)
    order = page.locator('select[name="order"]')
    shown_order = order.input_value()

    order.select_option("lot")
    expect(page).to_have_url(f"{fitted_server_url}/monitoring?order=lot")
    page.go_back()

    expect(page).to_have_url(url)
    expect(page.locator("#busy-overlay")).to_be_hidden()
    expect(order).to_have_value(shown_order)
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")
