"""Browser tests of the preprocessing and PCA screen while a fit runs."""

from __future__ import annotations

import json
import re

import executor_jobs
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

STEMS = ["run-1", "run-2", "run-10"]
# Long enough for every stage of the stand-in job and the polls after it.
FIT_TIMEOUT_MS = 20_000


def _open_fit_page(page: Page, url: str) -> None:
    """Select the fixture files and open the preprocessing and PCA screen."""
    response = page.request.post(
        f"{url}/catalog/selection", form={"stems": json.dumps(STEMS)}
    )
    assert response.ok
    page.goto(f"{url}/fit")


def _submit_fit(page: Page) -> None:
    """Run fit and wait until the overlay shows the run."""
    page.get_by_role("button", name="fit を実行").click()
    expect(page.locator("#busy-overlay")).to_have_class("busy-overlay busy-visible")


def test_groups_collapse(page: Page, slow_fit_server_url: str) -> None:
    """A settings group folds away and opens again from its heading."""
    _open_fit_page(page, slow_fit_server_url)
    group = page.locator('details[data-group="preprocess"]')
    wavelength = page.locator('input[name="wavelength_range_lower"]')
    expect(wavelength).to_be_visible()

    group.locator("summary").click()
    expect(wavelength).to_be_hidden()
    group.locator("summary").click()
    expect(wavelength).to_be_visible()


def test_running_fit_covers_the_page_until_it_ends(
    page: Page, slow_fit_server_url: str
) -> None:
    """The overlay shows the run's progress, survives a reload, and closes at the end."""
    _open_fit_page(page, slow_fit_server_url)
    _submit_fit(page)

    detail = page.locator("#busy-detail")
    expect(page.locator(".busy-spinner")).to_be_visible()
    expect(page.locator("#busy-message")).to_have_text("fit を実行しています…")
    expect(page.locator(".layout")).to_have_attribute("inert", "")
    expect(page.locator("#busy-cancel")).to_be_enabled()
    expect(detail).to_contain_text("段階 fit：1 / 4", timeout=FIT_TIMEOUT_MS)
    expect(detail).to_contain_text(re.compile(r"残り（全段階）：約 \d+ 秒"))

    # The reloaded page covers itself again and counts from the run's start.
    page.reload()
    expect(page.locator("#busy-overlay")).to_have_class("busy-overlay busy-visible")
    elapsed = int(page.locator("#busy-elapsed").text_content() or "0")
    assert elapsed >= 2

    expect(page.locator("#busy-overlay")).to_be_hidden(timeout=FIT_TIMEOUT_MS)
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")
    expect(page.locator("[data-run-status]")).to_have_text("failed")
    expect(page.locator("[data-run-error]")).to_contain_text(executor_jobs.SLOW_FIT_ERROR)
    expect(page.locator("[data-fit-runs] tbody tr")).to_have_count(1)


def test_cancel_stops_the_running_fit(page: Page, slow_fit_server_url: str) -> None:
    """Cancelling from the overlay cancels the run and then closes the overlay."""
    _open_fit_page(page, slow_fit_server_url)
    page.locator('details[data-group="status"] summary').click()
    _submit_fit(page)

    page.locator("#busy-cancel").click()

    expect(page.locator("#busy-overlay")).to_be_hidden(timeout=FIT_TIMEOUT_MS)
    expect(page.locator("[data-run-status]")).to_have_text("cancelled")
    # The closed status group opens to show the result.
    expect(page.locator('details[data-group="status"]')).to_have_attribute("open", "")
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")


def test_invalid_value_in_a_collapsed_group_opens_it(
    page: Page, slow_fit_server_url: str
) -> None:
    """The browser's validation opens the collapsed group of the invalid field."""
    _open_fit_page(page, slow_fit_server_url)
    group = page.locator('details[data-group="pca"]')
    n_component = page.locator('input[name="n_component"]')
    n_component.fill("0")
    group.locator("summary").click()
    expect(group).not_to_have_attribute("open", "")

    page.get_by_role("button", name="fit を実行").click()

    expect(group).to_have_attribute("open", "")
    expect(n_component).to_be_focused()
    expect(page.locator("#busy-overlay")).to_be_hidden()


def test_failed_cancel_request_can_be_retried(
    page: Page, slow_fit_server_url: str, expected_console_errors: list[str]
) -> None:
    """A cancel request that does not reach the server enables the button again."""
    expected_console_errors.append("net::ERR_FAILED")
    _open_fit_page(page, slow_fit_server_url)
    _submit_fit(page)
    page.route("**/runs/*/cancel", lambda route: route.abort())
    cancel = page.locator("#busy-cancel")

    cancel.click()

    expect(page.locator("#busy-message")).to_have_text(
        "キャンセルできませんでした。もう一度お試しください。"
    )
    expect(cancel).to_be_enabled()
    expect(page.locator("[data-run-status]")).to_have_text("running")

    page.unroute("**/runs/*/cancel")
    cancel.click()

    expect(page.locator("#busy-overlay")).to_be_hidden(timeout=FIT_TIMEOUT_MS)
    expect(page.locator("[data-run-status]")).to_have_text("cancelled")
