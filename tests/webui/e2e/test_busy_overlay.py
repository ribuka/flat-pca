"""Browser tests of the overlay shown while a page loads or its parts are replaced."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, Request, Route, expect
from sidebar_choice import (
    mark_page,
    select_in_dialog,
    shown_files,
    wait_for_sidebar,
    wait_for_view_refresh,
)

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
    order = page.locator('select[name="x_axis"]')
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
    order = page.locator('select[name="x_axis"]')
    shown_order = order.input_value()

    order.select_option("lot")
    expect(page).to_have_url(f"{fitted_server_url}/monitoring?color=lot&x_axis=lot")
    page.go_back()

    expect(page).to_have_url(url)
    expect(page.locator("#busy-overlay")).to_be_hidden()
    expect(order).to_have_value(shown_order)
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")


def _is_main_request(request: Request) -> bool:
    """Return whether a request fetches the main part of a screen."""
    return request.headers.get("hx-request") == "true" and "/explore" in request.url


def test_cancelling_a_refresh_shows_the_server_choice(
    page: Page, fitted_server_url: str
) -> None:
    """Cancelling the refresh after a choice keeps the main part and shows the choice."""
    pending: list[Route] = []

    def hold_main(route: Route) -> None:
        """Leave the request of the main part pending."""
        if _is_main_request(route.request):
            pending.append(route)
        else:
            route.continue_()

    page.route("**/explore**", hold_main)
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    heatmap = page.locator("#explore-heatmap")
    expect(heatmap).to_have_attribute("data-heatmap-label", "s-00")
    mark_page(page)

    select_in_dialog(page, ["s-05"])

    expect(page.locator("#busy-overlay")).to_have_class("busy-overlay busy-visible")
    page.locator("#busy-cancel").click()

    wait_for_view_refresh(page)
    expect(page.locator("#busy-overlay")).to_be_hidden()
    # The server took the choice before the cancel, and the sidebar shows it.
    expect(shown_files(page)).to_have_text(["s-05"])
    expect(heatmap).to_have_attribute("data-heatmap-label", "s-00")
    assert len(pending) == 1


def test_a_failed_refresh_releases_the_page(
    page: Page, fitted_server_url: str, expected_console_errors: list[str]
) -> None:
    """When the main part cannot be fetched, the overlay goes and nothing is replaced."""
    expected_console_errors.append("500")

    def fail_main(route: Route) -> None:
        """Answer the request of the main part with a server error."""
        if _is_main_request(route.request):
            route.fulfill(status=500, body="failed")
        else:
            route.continue_()

    page.route("**/explore**", fail_main)
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    mark_page(page)

    select_in_dialog(page, ["s-05"])

    wait_for_view_refresh(page)
    expect(page.locator("#busy-overlay")).to_be_hidden()
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-00")
