"""Browser tests of the transform screen: choosing targets and running transform."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import (
    mark_page,
    open_file_dialog,
    wait_for_sidebar,
    wait_for_view_refresh,
)

pytestmark = pytest.mark.e2e

CATALOG_STEMS = ["run-1", "run-2", "run-10"]
# Long enough for a transform job, or every stage of the stand-in job, and
# the polls after it.
TRANSFORM_TIMEOUT_MS = 20_000


def _expect_dialog_rows(page: Page, count: int) -> None:
    """Check the number of files the shown-file dialog offers, then cancel it."""
    dialog = open_file_dialog(page)
    expect(dialog.locator("tr[data-dt-key]")).to_have_count(count)
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_be_hidden()


def _choose_targets(page: Page) -> None:
    """Uncheck "use same data for fit" and choose every catalog file."""
    expect(page.locator("#files[data-dt-locked]")).to_have_count(1)
    mark_page(page)
    page.get_by_label("use same data for fit").uncheck()
    wait_for_view_refresh(page)
    expect(page.locator("#files[data-dt-locked]")).to_have_count(0)
    expect(page.locator("#files tbody tr")).to_have_count(len(CATALOG_STEMS))
    page.get_by_label("Select all filtered files").check()


def test_transform_page_locks_the_fit_targets_by_default(
    page: Page, fitted_server_url: str
) -> None:
    """Checked, the table and the button show, and the table cannot be edited."""
    page.goto(f"{fitted_server_url}/transform")
    wait_for_sidebar(page)

    expect(page.get_by_label("use same data for fit")).to_be_checked()
    expect(page.locator("[data-transform-model]")).to_have_value("fit-1")
    expect(page.locator("[data-transform-targets]")).to_be_visible()
    expect(page.locator("[data-fit-targets]")).to_contain_text("12 files")
    expect(page.locator("#files tbody tr")).to_have_count(len(CATALOG_STEMS))
    for box in page.locator("#files [data-dt-row-check]").all():
        expect(box).to_be_disabled()
    expect(page.get_by_label("Select all filtered files")).to_be_disabled()
    expect(page.get_by_role("button", name="Run transform")).to_be_enabled()
    expect(page.locator("[data-transform-shown]")).to_contain_text("tr-1")


def test_transform_runs_only_on_demand_and_once_per_data(
    page: Page, fitted_server_url: str
) -> None:
    """A transform of other files runs and is shown; the fit targets reuse ``tr-1``."""
    page.goto(f"{fitted_server_url}/transform")
    wait_for_sidebar(page)
    _choose_targets(page)

    page.get_by_role("button", name="Run transform").click()

    expect(page.locator("[data-run-status]")).to_have_text(
        "succeeded", timeout=TRANSFORM_TIMEOUT_MS
    )
    expect(page.locator("#busy-overlay")).to_be_hidden()
    rows = page.locator("[data-transform-runs] tbody tr")
    expect(rows).to_have_count(2)
    expect(rows.first.locator("td").nth(1)).to_have_text("fit-1")
    # The finished run is shown right away.
    expect(rows.first.locator("[data-shown-run]")).to_have_count(1)
    _expect_dialog_rows(page, len(CATALOG_STEMS))

    mark_page(page)
    rows.nth(1).get_by_role("button", name="Show").click()

    wait_for_view_refresh(page)
    expect(page.locator("[data-transform-shown]")).to_contain_text("tr-1")
    _expect_dialog_rows(page, 12)

    # Checked again, the fit targets were already transformed by tr-1.
    mark_page(page)
    page.get_by_label("use same data for fit").check()
    wait_for_view_refresh(page)
    page.get_by_role("button", name="Run transform").click()

    expect(page.locator('[data-transform-reused="tr-1"]')).to_be_visible()
    expect(rows).to_have_count(2)
    _expect_dialog_rows(page, 12)

    page.goto(f"{fitted_server_url}/monitoring")
    expect(page.locator("body")).not_to_contain_text("run-10")


def test_running_transform_covers_the_page_and_can_be_cancelled(
    page: Page, slow_transform_server_url: str
) -> None:
    """While a transform runs, the overlay shows it; cancelling ends the run."""
    page.goto(f"{slow_transform_server_url}/transform")
    wait_for_sidebar(page)
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


def test_help_tip_shows_on_hover_and_keyboard_focus(page: Page, fitted_server_url: str) -> None:
    """The explanation of "use same data for fit" shows only on hover or focus."""
    page.goto(f"{fitted_server_url}/transform")
    wait_for_sidebar(page)
    tip = page.locator("[data-transform-targets] #use-same-data-help")
    icon = page.locator('[aria-describedby="use-same-data-help"]')

    expect(icon).to_have_accessible_name("Help")
    expect(icon).to_have_accessible_description(
        "Checked, the transform targets are the files the model was fitted on. "
        "Unchecked, choose the transform targets from the catalog."
    )
    expect(tip).to_be_hidden()

    icon.hover()
    expect(tip).to_be_visible()
    expect(tip).to_contain_text("files the model was fitted on")
    page.mouse.move(0, 0)
    expect(tip).to_be_hidden()

    page.get_by_label("use same data for fit").focus()
    page.keyboard.press("Tab")
    expect(icon).to_be_focused()
    expect(tip).to_be_visible()
    page.keyboard.press("Escape")
    expect(tip).to_be_hidden()
    page.keyboard.press("Tab")
    expect(tip).to_be_hidden()
