"""Browser tests of the fit form's settings kept over a switch of screens."""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

STEMS = ["run-1", "run-2", "run-10"]


def _select_files(page: Page, url: str) -> None:
    """Select the fixture files as the fit target."""
    response = page.request.post(
        f"{url}/catalog/selection", form={"stems": json.dumps(STEMS)}
    )
    assert response.ok


def _switch_screens(page: Page) -> None:
    """Go to the data selection screen and back to the fit screen by the menu."""
    page.locator('a.nav-item[href="/"]').click()
    expect(page).to_have_title("Data selection - flat-pca")
    page.locator('a.nav-item[href="/fit"]').click()
    expect(page.locator("#fit-form")).to_be_visible()


def test_edited_settings_survive_a_switch_of_screens(
    page: Page, cataloged_server_url: str
) -> None:
    """Edited fields, selects, and checkboxes come back after another screen."""
    _select_files(page, cataloged_server_url)
    page.goto(f"{cataloged_server_url}/fit")
    form = page.locator("#fit-form")
    steps = form.locator('input[name="target_steps"]')
    expect(steps.first).to_be_checked()
    first_step = steps.first.get_attribute("value")
    enabled = form.locator('input[name="wavelength_range_enabled"]')
    was_enabled = enabled.is_checked()

    form.locator('input[name="n_component"]').fill("4")
    form.locator('input[name="edge_trim_start"]').fill("1.5")
    form.locator('select[name="scaling_strategy"]').select_option("z-score")
    steps.first.uncheck()
    enabled.set_checked(not was_enabled)

    _switch_screens(page)

    expect(form.locator('input[name="n_component"]')).to_have_value("4")
    expect(form.locator('input[name="edge_trim_start"]')).to_have_value("1.5")
    expect(form.locator('select[name="scaling_strategy"]')).to_have_value("z-score")
    expect(form.locator(f'input[name="target_steps"][value="{first_step}"]')).not_to_be_checked()
    if was_enabled:
        expect(enabled).not_to_be_checked()
    else:
        expect(enabled).to_be_checked()


def test_reopened_run_settings_become_the_kept_ones(
    page: Page, fitted_server_url: str
) -> None:
    """A reopened run replaces the edits, and its settings are kept in turn."""
    _select_files(page, fitted_server_url)
    page.goto(f"{fitted_server_url}/fit")
    n_component = page.locator('#fit-form input[name="n_component"]')
    n_component.fill("5")

    page.goto(f"{fitted_server_url}/fit?run=fit-1")
    expect(n_component).to_have_value("3")
    expect(page.locator('#fit-form select[name="impute_strategy"]')).to_have_value("median")

    _switch_screens(page)

    expect(n_component).to_have_value("3")
    expect(page.locator('#fit-form select[name="impute_strategy"]')).to_have_value("median")


def test_a_new_tab_shows_the_defaults(page: Page, cataloged_server_url: str) -> None:
    """The settings are kept per tab, so another tab starts from the defaults."""
    _select_files(page, cataloged_server_url)
    page.goto(f"{cataloged_server_url}/fit")
    n_component = page.locator('#fit-form input[name="n_component"]')
    default = n_component.input_value()
    n_component.fill("4")

    other = page.context.new_page()
    other.goto(f"{cataloged_server_url}/fit")

    expect(other.locator('#fit-form input[name="n_component"]')).to_have_value(default)
    other.close()
