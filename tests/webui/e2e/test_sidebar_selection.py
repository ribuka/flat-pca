"""Browser tests of the shown run and files and of the transform page's choices."""

from __future__ import annotations

import pytest
from datatable_menu import close_column_menu, open_column_menu
from playwright.sync_api import Page, Request, expect
from sidebar_choice import (
    choose_files,
    mark_page,
    open_file_dialog,
    select_in_dialog,
    shown_files,
    wait_for_sidebar,
    wait_for_view_refresh,
)

pytestmark = pytest.mark.e2e


def test_files_chosen_in_the_dialog_follow_every_screen_and_reload(
    page: Page, fitted_server_url: str
) -> None:
    """Files selected in the dialog replace the main part and are kept elsewhere."""
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    expect(shown_files(page)).to_have_count(0)
    expect(page.locator("[data-view-file-count]")).to_have_text("0 / 20")
    expect(page.locator("#view-selection input")).to_have_count(0)

    dialog = open_file_dialog(page)
    rows = dialog.locator("tr[data-dt-key]")
    expect(rows).to_have_count(12)
    expect(rows.first).to_have_attribute("data-dt-key", "s-00")
    menu = open_column_menu(dialog, "file")
    menu.get_by_label("Filter by file name").fill("s-1")
    expect(rows).to_have_count(2)
    # The menu shows on top of the dialog; Escape closes it but not the dialog.
    expect(menu).to_be_visible()
    box = menu.bounding_box()
    assert box is not None
    corners = [[box["x"] + 5, box["y"] + 5], [box["x"] + 5, box["y"] + box["height"] - 5]]
    assert menu.evaluate(
        "(menu, corners) => corners.every(([x, y]) => menu.contains(document.elementFromPoint(x, y)))",
        corners,
    )
    close_column_menu(page, menu)
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("button", name="file", exact=True)).to_be_focused()
    # Without the clipboard API (plain HTTP from another host) the name is
    # still copied from inside the modal dialog.
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.evaluate(
        "() => { window.realClipboard = navigator.clipboard;"
        " Object.defineProperty(navigator, 'clipboard', {value: undefined, configurable: true}); }"
    )
    open_column_menu(dialog, "lot").get_by_role("button", name="Copy column name").click()
    assert page.evaluate("() => window.realClipboard.readText()") == "lot"
    open_column_menu(dialog, "file").get_by_role("button", name="Desc").click()
    expect(rows).to_have_count(2)
    expect(rows.first).to_have_attribute("data-dt-key", "s-11")
    dialog.get_by_label("Select s-10", exact=True).check()
    expect(dialog.locator("[data-dt-selected-count]")).to_have_text("1 / 20 selected")
    mark_page(page)
    dialog.get_by_role("button", name="Select", exact=True).click()

    expect(dialog).to_be_hidden()
    wait_for_view_refresh(page)
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-10")
    expect(shown_files(page)).to_have_text(["s-10"])
    expect(page.locator("[data-view-file-count]")).to_have_text("1 / 20")
    # A closed dialog keeps no table in the page.
    expect(page.locator("#view-files")).to_have_count(0)

    page.locator('a.nav-item[href="/scores"]').click()
    expect(page.locator("#scores-trajectories")).to_have_attribute("data-plot-ready", "true")
    expect(shown_files(page)).to_have_text(["s-10"])
    expect(page.locator("#scores-trajectories .legend .traces").nth(0)).to_contain_text(
        "s-10"
    )

    page.reload()
    expect(shown_files(page)).to_have_text(["s-10"])


def test_closing_without_select_keeps_the_choice(page: Page, fitted_server_url: str) -> None:
    """Cancel, Esc, and the close button save nothing; reopening starts from the choice."""
    choose_files(page, fitted_server_url, ["s-03"])
    page.goto(f"{fitted_server_url}/explore")
    wait_for_sidebar(page)
    expect(shown_files(page)).to_have_text(["s-03"])
    posts: list[str] = []
    page.on(
        "request",
        lambda request: posts.append(request.url) if request.method == "POST" else None,
    )

    for close in (
        lambda dialog: dialog.get_by_role("button", name="Cancel").click(),
        lambda dialog: page.keyboard.press("Escape"),
        lambda dialog: dialog.get_by_role("button", name="Close").click(),
    ):
        dialog = open_file_dialog(page)
        expect(dialog.get_by_label("Select s-03", exact=True)).to_be_checked()
        expect(dialog.get_by_label("Select s-04", exact=True)).not_to_be_checked()
        expect(dialog.locator("[data-dt-selected-count]")).to_have_text("1 / 20 selected")
        dialog.get_by_label("Select s-03", exact=True).uncheck()
        dialog.get_by_label("Select s-04", exact=True).check()

        close(dialog)

        expect(dialog).to_be_hidden()
        expect(shown_files(page)).to_have_text(["s-03"])
    assert posts == []
    expect(page.locator("#explore-heatmap")).to_have_attribute("data-heatmap-label", "s-03")


def test_the_dialog_stops_at_the_limit(page: Page, one_file_server_url: str) -> None:
    """At ``ui.explore_max_files`` the other rows cannot be checked."""
    page.goto(f"{one_file_server_url}/explore")
    wait_for_sidebar(page)
    dialog = open_file_dialog(page)
    notice = dialog.locator("[data-dt-limit]")
    expect(notice).to_be_hidden()

    dialog.get_by_label("Select s-01", exact=True).check()

    expect(notice).to_be_visible()
    expect(dialog.get_by_label("Select s-02", exact=True)).to_be_disabled()
    expect(dialog.locator("[data-dt-selected-count]")).to_have_text("1 / 1 selected")
    mark_page(page)
    dialog.get_by_role("button", name="Select", exact=True).click()
    wait_for_view_refresh(page)
    expect(shown_files(page)).to_have_text(["s-01"])
    expect(page.locator("[data-view-file-count]")).to_have_text("1 / 1")

    dialog = open_file_dialog(page)
    expect(dialog.get_by_label("Select s-00", exact=True)).to_be_disabled()
    expect(dialog.get_by_label("Select s-01", exact=True)).to_be_checked()


def test_changing_the_run_clears_the_files(page: Page, two_fits_server_url: str) -> None:
    """The newest transform run is shown by default; showing another clears the files."""
    page.goto(f"{two_fits_server_url}/transform")
    wait_for_sidebar(page)
    expect(page.locator("#view-selection select")).to_have_count(0)
    shown = page.locator("[data-transform-shown]")
    expect(shown).to_have_attribute("data-transform-shown", "tr-2")
    mark_page(page)
    select_in_dialog(page, ["s-00"])
    wait_for_view_refresh(page)
    expect(shown_files(page)).to_have_text(["s-00"])

    page.locator('[data-run-id="tr-1"]').get_by_role("button", name="Show").click()

    wait_for_view_refresh(page)
    expect(shown).to_have_attribute("data-transform-shown", "tr-1")
    expect(shown_files(page)).to_have_count(0)
    dialog = open_file_dialog(page)
    expect(dialog.locator("[data-view-dialog-run]")).to_have_text("run tr-1")
    expect(dialog.locator("tr[data-dt-key]")).to_have_count(12)
    expect(dialog.locator("[data-dt-row-check]:checked")).to_have_count(0)


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


def test_repeated_refreshes_do_not_repeat_the_trend_requests(
    page: Page, fitted_server_url: str
) -> None:
    """After several replacements, one slider move fetches the trends once."""
    page.goto(f"{fitted_server_url}/explore?view=raw&segment=1:1")
    wait_for_sidebar(page)
    mark_page(page)
    for stem in ("s-00", "s-01", "s-02"):
        select_in_dialog(page, [stem])
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

        select_in_dialog(page, ["s-03"])

        wait_for_view_refresh(page)
        expect(shown_files(page)).to_have_text(["s-03"])
        expect(page.locator("main .card").first).to_have_attribute("data-kept", "true")
        select_in_dialog(page, ["s-03"])
        wait_for_view_refresh(page)
        expect(shown_files(page)).to_have_count(0)


def test_a_stale_dialog_shows_the_run_in_use(page: Page, two_fits_server_url: str) -> None:
    """Files selected in a dialog opened before the run changed elsewhere show the new run."""
    page.goto(f"{two_fits_server_url}/explore")
    wait_for_sidebar(page)
    expect(page.locator("[data-shown-files]")).to_contain_text("run tr-2")
    dialog = open_file_dialog(page)
    # Another tab shows the other run.
    response = page.request.post(
        f"{two_fits_server_url}/transform/show",
        form={"run": "tr-1"},
    )
    assert response.ok
    mark_page(page)

    dialog.get_by_label("Select s-00", exact=True).check()
    dialog.get_by_role("button", name="Select", exact=True).click()

    wait_for_view_refresh(page)
    expect(page.locator("[data-shown-files]")).to_contain_text("run tr-1")
    expect(shown_files(page)).to_have_count(0)
