"""Browser tests of the model page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

# Segment (2, 1) of the synthetic spectra: StepTime 0, 1, 2 over five wavelengths.
MODEL_PATH = "/model?segment=2:1"


def _open(page: Page, url: str) -> None:
    """Open a model page and wait until its figures and initial trends are drawn."""
    page.goto(url)
    for target in ("#model-scree", "#model-loadings"):
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "402.5")
    expect(root).to_have_attribute("data-trend-step-time", "1")


def test_slider_moves_the_component_trends(page: Page, fitted_server_url: str) -> None:
    """The StepTime slider cuts the component at another time point."""
    _open(page, fitted_server_url + MODEL_PATH)
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "PC1")
    expect(page.locator("#explore-heatmap .shapelayer path")).to_have_count(2)

    page.locator("#explore-step-time").press("End")

    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-step-time", "2")
    expect(page.locator("#explore-trend-wavelength .gtitle")).to_have_text("StepTime = 2")


def test_choices_reload_the_page(page: Page, fitted_server_url: str) -> None:
    """Choosing another component k and aggregation reloads the page with them."""
    _open(page, fitted_server_url + MODEL_PATH)

    k_input = page.locator('input[name="k"]')
    k_input.fill("3")
    k_input.dispatch_event("change")

    expect(page).to_have_url(
        f"{fitted_server_url}/model?x=1&y=2&aggregation=rms&view=component&k=3&segment=2%3A1"
    )
    _open(page, page.url)
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "PC3")

    page.locator('select[name="aggregation"]').select_option("abs_mean")

    expect(page).to_have_url(
        f"{fitted_server_url}/model?x=1&y=2&aggregation=abs_mean&view=component&k=3&segment=2%3A1"
    )
    expect(page.locator("#model-loadings")).to_have_attribute("data-plot-ready", "true")
    expect(page.locator("#model-loadings .gtitle")).to_have_text("ローディング（絶対値平均）")
