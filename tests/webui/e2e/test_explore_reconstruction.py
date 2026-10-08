"""Browser tests of the reconstruction views of the spectral exploration page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import choose_files
from spectra import SPECTRA_SHORT_FILE

pytestmark = pytest.mark.e2e

SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"
# Segment (2, 1) of the synthetic spectra: StepTime 0, 1, 2 over five wavelengths.
RECONSTRUCTION_PATH = "/explore?view=reconstruction&segment=2:1&k=2"


def _open(page: Page, base_url: str) -> None:
    """Open the reconstruction view of the short file and wait for the initial trends."""
    choose_files(page, base_url, [SHORT])
    page.goto(base_url + RECONSTRUCTION_PATH)
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "402.5")
    expect(root).to_have_attribute("data-trend-step-time", "1")


def test_reconstruction_trends_overlay_the_preprocessed_values(
    page: Page, fitted_server_url: str
) -> None:
    """The trends draw the reconstruction and the preprocessed values together."""
    _open(page, fitted_server_url)

    expect(page.locator("#explore-heatmap")).to_have_attribute(
        "data-heatmap-label", f"{SHORT} (reconstruction)"
    )
    legend = page.locator("#explore-trend-step-time .legend .traces")
    expect(legend).to_have_count(2)
    expect(legend.nth(0)).to_contain_text("(reconstruction)")
    expect(legend.nth(1)).to_contain_text("(preprocessed)")


def test_view_and_component_choices_reload_the_page(
    page: Page, fitted_server_url: str
) -> None:
    """Choosing the residual view and another k keeps the file and segment."""
    _open(page, fitted_server_url)

    page.locator('select[name="view"]').select_option("residual")
    expect(page).to_have_url(
        f"{fitted_server_url}/explore?view=residual&segment=2%3A1&k=2&heatmap_file={SHORT}"
    )
    expect(page.locator("[data-view-note]")).to_contain_text("1..2 成分")

    k_input = page.locator('input[name="k"]')
    k_input.fill("3")
    k_input.dispatch_event("change")

    expect(page.locator("[data-view-note]")).to_contain_text("1..3 成分")
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", SHORT)
    expect(page.locator("#explore-trend-step-time .legend .traces")).to_have_count(1)
