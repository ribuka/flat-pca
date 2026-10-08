"""Browser tests of the sidebar's run and file choices and the transform page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import wait_for_sidebar

pytestmark = pytest.mark.e2e


def test_chosen_files_follow_every_screen_and_reload(
    page: Page, fitted_server_url: str
) -> None:
    """Shown files chosen on one screen are kept on the others and after a reload."""
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    items = page.locator("#view-selection li[data-stem]")
    expect(items).to_have_count(12)
    expect(items.nth(0)).to_have_attribute("data-stem", "s-00")

    page.locator("[data-view-file-search]").fill("s-1")

    expect(page.locator("#view-selection li[data-stem]:visible")).to_have_count(2)
    with page.expect_navigation():
        page.locator('#view-selection input[value="s-10"]').check()

    wait_for_sidebar(page)
    expect(page.locator("[data-view-file-search]")).to_have_value("s-1")
    expect(page.locator("#view-selection li[data-stem]:visible")).to_have_count(2)
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-10")
    expect(page.locator("[data-view-file-count]")).to_have_text("1 / 20")

    page.locator('a.nav-item[href="/scores"]').click()
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    expect(page.locator('#view-selection input[value="s-10"]')).to_be_checked()
    expect(page.locator("#scores-trajectories .legend .traces").nth(0)).to_contain_text(
        "s-10"
    )

    page.reload()
    expect(page.locator('#view-selection input[value="s-10"]')).to_be_checked()


def test_changing_the_run_clears_the_files(page: Page, two_fits_server_url: str) -> None:
    """The newest run is used by default; choosing another run clears the files."""
    page.goto(f"{two_fits_server_url}/model")
    wait_for_sidebar(page)
    run = page.locator("#view-selection select[name=run]")
    expect(run).to_have_value("fit-2")
    expect(page.locator("main")).to_contain_text("run: fit-2")
    with page.expect_navigation():
        page.locator('#view-selection input[value="s-00"]').check()
    wait_for_sidebar(page)
    expect(page.locator('#view-selection input[value="s-00"]')).to_be_checked()

    with page.expect_navigation():
        run.select_option("fit-1")

    wait_for_sidebar(page)
    expect(run).to_have_value("fit-1")
    expect(page.locator("main")).to_contain_text("run: fit-1")
    expect(page.locator("#view-selection input[name=file]")).to_have_count(12)
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(0)


def test_transform_page_uses_the_fit_data_by_default(
    page: Page, fitted_server_url: str
) -> None:
    """The transform page follows the preprocessing page and starts on the fit data."""
    page.goto(f"{fitted_server_url}/fit")

    page.locator('a.nav-item[href="/transform"]').click()

    checkbox = page.get_by_label("use same data for fit")
    expect(checkbox).to_be_checked()
    expect(checkbox).to_be_enabled()
    expect(page.locator("[data-transform-targets]")).to_be_hidden()
    expect(page.locator("[data-transform-shown]")).to_contain_text("12 files")
