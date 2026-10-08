"""Browser tests of the transform screen: choosing targets and running transform."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import mark_page, wait_for_sidebar, wait_for_view_refresh

pytestmark = pytest.mark.e2e

CATALOG_STEMS = ["run-1", "run-2", "run-10"]
# Long enough for a transform job, or every stage of the stand-in job, and
# the polls after it.
TRANSFORM_TIMEOUT_MS = 20_000


def _choose_targets(page: Page) -> None:
    """Uncheck "use same data for fit" and choose every catalog file."""
    targets = page.locator("[data-transform-targets]")
    expect(targets).to_be_hidden()
    page.get_by_label("use same data for fit").uncheck()
    expect(targets).to_be_visible()
    expect(page.locator("#file-table tbody tr")).to_have_count(len(CATALOG_STEMS))
    page.get_by_label("Select all filtered files").check()


def test_transform_runs_and_its_targets_are_shown(
    page: Page, fitted_server_url: str
) -> None:
    """A transform of other files can be run, shown in the sidebar, and switched back."""
    page.goto(f"{fitted_server_url}/transform")
    wait_for_sidebar(page)
    expect(page.get_by_label("use same data for fit")).to_be_checked()
    _choose_targets(page)

    page.get_by_role("button", name="Run transform").click()

    expect(page.locator("[data-run-status]")).to_have_text(
        "succeeded", timeout=TRANSFORM_TIMEOUT_MS
    )
    expect(page.locator("#busy-overlay")).to_be_hidden()
    rows = page.locator("[data-transform-runs] tbody tr")
    expect(rows).to_have_count(1)
    expect(rows.first.locator("td").nth(1)).to_have_text("fit-1")
    # The finished run is offered in the sidebar right away.
    expect(page.locator('#view-run option[data-run-kind="transform"]')).to_have_count(1)

    mark_page(page)
    rows.first.get_by_role("button", name="Show").click()

    wait_for_view_refresh(page)
    expect(page.locator("#view-run option:checked")).to_have_attribute(
        "data-run-kind", "transform"
    )
    expect(page.locator("#view-selection li[data-stem]")).to_have_count(
        len(CATALOG_STEMS)
    )
    expect(page.get_by_label("use same data for fit")).not_to_be_checked()
    expect(page.locator("[data-transform-targets]")).to_be_visible()
    expect(page.locator("[data-shown-run]")).to_have_count(1)

    # Checking the box shows the fit run's own data again.
    page.get_by_label("use same data for fit").check()

    wait_for_view_refresh(page)
    expect(page.locator("#view-run option:checked")).to_have_attribute("value", "fit-1")
    expect(page.get_by_label("use same data for fit")).to_be_checked()
    expect(page.locator("[data-transform-targets]")).to_be_hidden()

    page.goto(f"{fitted_server_url}/monitoring")
    expect(page.locator("body")).not_to_contain_text("run-10")


def test_running_transform_covers_the_page_and_can_be_cancelled(
    page: Page, slow_transform_server_url: str
) -> None:
    """While a transform runs, the overlay shows it; cancelling ends the run."""
    page.goto(f"{slow_transform_server_url}/transform")
    _choose_targets(page)

    page.get_by_role("button", name="Run transform").click()

    expect(page.locator("#busy-overlay")).to_have_class("busy-overlay busy-visible")
    expect(page.locator("#busy-message")).to_have_text("Running the transform…")
    expect(page.locator("#busy-detail")).to_contain_text(
        "Stage", timeout=TRANSFORM_TIMEOUT_MS
    )

    page.locator("#busy-cancel").click()

    expect(page.locator("#busy-overlay")).to_be_hidden(timeout=TRANSFORM_TIMEOUT_MS)
    expect(page.locator("[data-run-status]")).to_have_text("cancelled")
    expect(page.locator("[data-transform-runs] tbody tr")).to_have_count(1)
