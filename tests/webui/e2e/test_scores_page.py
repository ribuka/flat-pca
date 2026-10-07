"""Browser tests of the score page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import choose_files, wait_for_sidebar
from spectra import SPECTRA_SHORT_FILE

pytestmark = pytest.mark.e2e

SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"


def _open(page: Page, url: str) -> None:
    """Open a score page and wait until the score scatter plot is drawn."""
    page.goto(url)
    expect(page.locator("#scores-scatter")).to_have_attribute("data-plot-ready", "true")
    wait_for_sidebar(page)


def _click_point(page: Page, index: int) -> None:
    """Click one point of the score scatter plot."""
    # The plot's drag layer covers the points, so click at the point's position.
    box = page.locator("#scores-scatter .scatterlayer .point").nth(index).bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_clicking_a_score_point_adds_the_file(page: Page, fitted_server_url: str) -> None:
    """A click on a score point adds the file to the sidebar and stays on the page."""
    url = f"{fitted_server_url}/scores?color="
    _open(page, url)
    expect(page.locator("#view-selection input[name=file]")).to_have_count(12)

    with page.expect_navigation():
        _click_point(page, 1)

    expect(page).to_have_url(url)
    expect(page.locator('#view-selection input[value="s-01"]')).to_be_checked()
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(1)
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(1)
    expect(legend.nth(0)).to_contain_text("s-01")


def test_clicking_a_point_at_the_limit_warns(page: Page, one_file_server_url: str) -> None:
    """At the limit of shown files, a clicked file is not added and a warning is shown."""
    choose_files(page, one_file_server_url, ["s-00"])
    url = f"{one_file_server_url}/scores?color="
    _open(page, url)
    expect(page.locator('#view-selection input[value="s-00"]')).to_be_checked()

    _click_point(page, 1)

    warning = page.locator("#plot-select-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("1 件まで")
    expect(warning).to_contain_text("s-01")
    assert page.url == url
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(1)


def test_choices_redraw_the_trajectories(page: Page, fitted_server_url: str) -> None:
    """Files chosen in the sidebar and another component redraw the trajectories."""
    _open(page, f"{fitted_server_url}/scores")
    expect(page.locator("#scores-trajectories .legend .traces")).to_have_count(1)
    expect(page.locator('select[name="file"]')).to_have_count(0)
    expect(page.locator('select[name="run"]')).to_have_count(1)

    for stem in ("s-00", SHORT):
        with page.expect_navigation():
            page.locator(f'#view-selection input[value="{stem}"]').check()
        _open(page, page.url)
    y_input = page.locator('input[name="y"]')
    y_input.fill("3")
    y_input.dispatch_event("change")

    expect(page).to_have_url(f"{fitted_server_url}/scores?x=1&y=3&color=lot")
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(2)
    expect(legend.nth(1)).to_contain_text(SHORT)
    expect(page.locator("#scores-trajectories .xtitle")).to_have_text("PC1")
    expect(page.locator("#scores-trajectories .ytitle")).to_have_text("PC3")


def test_a_failed_click_request_releases_the_page(
    page: Page, fitted_server_url: str
) -> None:
    """When the request of a clicked point fails, the overlay goes and a warning shows."""
    url = f"{fitted_server_url}/scores?color="
    _open(page, url)
    page.evaluate("() => { window.fetch = () => Promise.reject(new TypeError('offline')); }")

    _click_point(page, 1)

    warning = page.locator("#plot-select-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("s-01 を表示ファイルに追加できませんでした")
    expect(page.locator("#busy-overlay")).to_be_hidden()
    expect(page.locator(".layout")).not_to_have_attribute("inert", "")
    assert page.url == url
