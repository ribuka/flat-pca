"""Browser tests of the shown run and files and of the transform page's choices."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, Request, expect
from sidebar_choice import mark_page, wait_for_sidebar, wait_for_view_refresh

pytestmark = pytest.mark.e2e


def test_chosen_files_follow_every_screen_and_reload(
    page: Page, fitted_server_url: str
) -> None:
    """Shown files chosen on one screen replace its main part and are kept elsewhere."""
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    items = page.locator("#view-selection li[data-stem]")
    expect(items).to_have_count(12)
    expect(items.nth(0)).to_have_attribute("data-stem", "s-00")

    page.locator("[data-view-file-search]").fill("s-1")

    expect(page.locator("#view-selection li[data-stem]:visible")).to_have_count(2)
    mark_page(page)
    page.locator('#view-selection input[value="s-10"]').check()

    wait_for_view_refresh(page)
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
    """The newest transform run is shown by default; showing another clears the files."""
    page.goto(f"{two_fits_server_url}/transform")
    wait_for_sidebar(page)
    expect(page.locator("#view-selection select")).to_have_count(0)
    shown = page.locator("[data-transform-shown]")
    expect(shown).to_have_attribute("data-transform-shown", "tr-2")
    mark_page(page)
    page.locator('#view-selection input[value="s-00"]').check()
    wait_for_view_refresh(page)
    expect(page.locator('#view-selection input[value="s-00"]')).to_be_checked()

    page.locator('[data-run-id="tr-1"]').get_by_role("button", name="Show").click()

    wait_for_view_refresh(page)
    expect(shown).to_have_attribute("data-transform-shown", "tr-1")
    expect(page.locator("#view-selection input[name=file]")).to_have_count(12)
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(0)


def test_model_choice_replaces_the_transform_page(page: Page, two_fits_server_url: str) -> None:
    """The model is chosen on the transform page; the model page follows it."""
    page.goto(f"{two_fits_server_url}/transform")
    wait_for_sidebar(page)
    model = page.locator("[data-transform-model]")
    expect(model).to_have_value("fit-2")
    mark_page(page)

    model.select_option("fit-1")

    wait_for_view_refresh(page)
    expect(model).to_have_value("fit-1")
    page.locator('a.nav-item[href="/model"]').click()
    expect(page.locator("[data-model-run]")).to_have_attribute("data-model-run", "fit-1")


def test_choosing_a_file_keeps_the_list_scroll(page: Page, fitted_server_url: str) -> None:
    """The shown-file list keeps its scroll position over the refresh."""
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    # A short list scrolls with the 12 files of the fixture.
    page.add_style_tag(content=".view-file-list { max-height: 5rem; }")
    file_list = page.locator("#view-selection .view-file-list")
    scrolled = file_list.evaluate(
        "(list) => { list.scrollTop = list.scrollHeight; return list.scrollTop; }"
    )
    assert scrolled > 0
    mark_page(page)

    page.locator('#view-selection input[value="s-11"]').check()

    wait_for_view_refresh(page)
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-11")
    assert file_list.evaluate("(list) => list.scrollTop") == scrolled


def test_repeated_refreshes_do_not_repeat_the_trend_requests(
    page: Page, fitted_server_url: str
) -> None:
    """After several replacements, one slider move fetches the trends once."""
    page.goto(f"{fitted_server_url}/explore?view=raw&segment=1:1")
    wait_for_sidebar(page)
    mark_page(page)
    for stem in ("s-00", "s-01", "s-02"):
        page.locator(f'#view-selection input[value="{stem}"]').check()
        wait_for_view_refresh(page)
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-wavelength", "402.5")
    expect(page.locator("#explore-trend-step-time .scatterlayer .trace")).to_have_count(3)
    expect(page.locator("main .js-plotly-plot")).to_have_count(3)
    trend_requests: list[str] = []

    def record(request: Request) -> None:
        """Record a request of the trends."""
        if "/explore/trend" in request.url:
            trend_requests.append(request.url)

    page.on("request", record)

    page.locator("#explore-wavelength").press("ArrowRight")

    expect(root).to_have_attribute("data-trend-wavelength", "405")
    # Requests of figures drawn before a replacement would follow the debounce.
    page.wait_for_timeout(500)
    assert len(trend_requests) == 1


def test_screens_without_the_choice_keep_their_main_part(
    page: Page, fitted_server_url: str
) -> None:
    """Data selection keeps its main part for any choice; transform for the files."""
    for path in ("/", "/transform"):
        page.goto(f"{fitted_server_url}{path}")
        wait_for_sidebar(page)
        mark_page(page)
        page.locator("main .card").first.evaluate("(card) => { card.dataset.kept = 'true'; }")

        page.locator('#view-selection input[value="s-03"]').check()

        wait_for_view_refresh(page)
        expect(page.locator('#view-selection input[value="s-03"]')).to_be_checked()
        expect(page.locator("main .card").first).to_have_attribute("data-kept", "true")
        page.locator('#view-selection input[value="s-03"]').uncheck()
        wait_for_view_refresh(page)


def test_a_stale_sidebar_shows_the_run_in_use(page: Page, two_fits_server_url: str) -> None:
    """A file chosen in a sidebar drawn before the run changed elsewhere shows the new run."""
    page.goto(f"{two_fits_server_url}/explore")
    wait_for_sidebar(page)
    expect(page.locator("[data-shown-files]")).to_contain_text("run tr-2")
    # Another tab shows the other run.
    response = page.request.post(
        f"{two_fits_server_url}/transform/show",
        form={"run": "tr-1"},
    )
    assert response.ok
    mark_page(page)

    page.locator('#view-selection input[value="s-00"]').check()

    wait_for_view_refresh(page)
    expect(page.locator("[data-shown-files]")).to_contain_text("run tr-1")
    expect(page.locator("#view-selection input[name=file]:checked")).to_have_count(0)
