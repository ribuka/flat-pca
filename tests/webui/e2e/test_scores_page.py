"""Browser tests of the score and loading page."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect
from spectra import SPECTRA_SHORT_FILE

pytestmark = pytest.mark.e2e

SHORT = f"s-{SPECTRA_SHORT_FILE:02d}"


def _open(page: Page, url: str) -> None:
    """Open a score page and wait until every figure is drawn."""
    page.goto(url)
    for target in ("#scores-scatter", "#scores-loadings", "#scores-scree"):
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")


def test_clicking_a_score_point_opens_the_exploration(
    page: Page, fitted_server_url: str
) -> None:
    """A click on a score point opens that file's preprocessed spectra."""
    _open(page, f"{fitted_server_url}/scores?color=")

    # The plot's drag layer covers the points, so click at the point's position.
    box = page.locator("#scores-scatter .scatterlayer .point").first.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    expect(page).to_have_url(
        f"{fitted_server_url}/explore?view=preprocessed&run=fit-1&file=s-00"
    )
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-00")


def test_choices_redraw_the_trajectories(page: Page, fitted_server_url: str) -> None:
    """Choosing files and another component reloads the page with them."""
    _open(page, f"{fitted_server_url}/scores")
    expect(page.locator("#scores-trajectories .legend .traces")).to_have_count(1)

    page.locator('select[name="file"]').select_option(["s-00", SHORT])
    expect(page).to_have_url(re.compile(rf"&file=s-00&file={SHORT}$"))
    _open(page, page.url)
    y_input = page.locator('input[name="y"]')
    y_input.fill("3")
    y_input.dispatch_event("change")

    expect(page).to_have_url(
        f"{fitted_server_url}/scores?run=fit-1&x=1&y=3&color=lot&aggregation=rms"
        f"&file=s-00&file={SHORT}"
    )
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    legend = page.locator("#scores-trajectories .legend .traces")
    expect(legend).to_have_count(2)
    expect(legend.nth(1)).to_contain_text(SHORT)
    expect(page.locator("#scores-trajectories .xtitle")).to_have_text("PC1")
    expect(page.locator("#scores-trajectories .ytitle")).to_have_text("PC3")
