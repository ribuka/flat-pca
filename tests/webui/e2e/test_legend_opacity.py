"""Browser tests of the legend fading traces instead of hiding them."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from sidebar_choice import choose_files

pytestmark = pytest.mark.e2e

INACTIVE_OPACITY = 0.1
# Longer than Plotly's double-click delay (300 ms), after which a legend click acts.
DOUBLE_CLICK_DELAY_MS = 400


def _opacities(page: Page, target: str) -> list[float]:
    """Return the opacity of each trace of a figure, 1 when it is not set."""
    return page.evaluate(
        "(target) => document.querySelector(target).data.map((trace) => trace.opacity ?? 1)",
        target,
    )


def _legend_item(page: Page, target: str, index: int):
    """Return the click area of a legend item of a figure."""
    return page.locator(f"{target} .legend .traces .legendtoggle").nth(index)


def _click_legend(page: Page, target: str, index: int, button: str = "left") -> None:
    """Click a legend item and wait until Plotly has handled the click."""
    _legend_item(page, target, index).click(button=button)
    page.wait_for_timeout(DOUBLE_CLICK_DELAY_MS)


def _open_scores(page: Page, url: str) -> None:
    """Open a score page and wait until both figures are drawn."""
    page.goto(url)
    for target in ("#scores-scatter", "#scores-trajectories"):
        expect(page.locator(target)).to_have_attribute("data-plot-ready", "true")


def test_legend_click_fades_the_trace(page: Page, fitted_server_url: str) -> None:
    """A legend click fades a trace instead of hiding it, and a second one restores it."""
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    _open_scores(page, f"{fitted_server_url}/scores")
    target = "#scores-trajectories"

    _click_legend(page, target, 1)

    assert _opacities(page, target) == [1, INACTIVE_OPACITY, 1]
    visible = page.evaluate(
        "(target) => document.querySelector(target).data.map((trace) => trace.visible ?? true)",
        target,
    )
    assert visible == [True, True, True]
    expect(page.locator(f"{target} .cartesianlayer .trace.scatter")).to_have_count(3)

    _click_legend(page, target, 1)

    assert _opacities(page, target) == [1, 1, 1]


def _drawn_order(page: Page, target: str) -> list[int]:
    """Return the trace indices of a figure's scatter traces in drawing order."""
    return page.evaluate(
        """(target) => [...document.querySelectorAll(`${target} .cartesianlayer .trace.scatter`)]
          .map((node) => node.__data__[0].trace.index)""",
        target,
    )


def test_inactive_trace_is_drawn_behind(page: Page, fitted_server_url: str) -> None:
    """An inactive trace is drawn before, so behind, the active traces."""
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    _open_scores(page, f"{fitted_server_url}/scores")
    target = "#scores-trajectories"
    assert _drawn_order(page, target) == [0, 1, 2]

    _click_legend(page, target, 2)

    assert _drawn_order(page, target) == [2, 0, 1]

    _click_legend(page, target, 2)

    assert _drawn_order(page, target) == [0, 1, 2]


def test_legend_right_click_isolates_the_trace(page: Page, fitted_server_url: str) -> None:
    """A right click makes only that trace active; again, it makes every trace active."""
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    _open_scores(page, f"{fitted_server_url}/scores")
    target = "#scores-trajectories"
    # The browser shows its menu for a context menu event that is not cancelled.
    page.evaluate(
        """() => {
          window.legendMenuShown = false;
          document.addEventListener("contextmenu", (event) => {
            window.legendMenuShown ||= !event.defaultPrevented;
          });
        }"""
    )

    _click_legend(page, target, 2, button="right")

    assert _opacities(page, target) == [INACTIVE_OPACITY, INACTIVE_OPACITY, 1]
    assert page.evaluate("() => window.legendMenuShown") is False

    _click_legend(page, target, 0, button="right")

    assert _opacities(page, target) == [1, INACTIVE_OPACITY, INACTIVE_OPACITY]

    _click_legend(page, target, 0, button="right")

    assert _opacities(page, target) == [1, 1, 1]


def test_legend_double_click_changes_nothing(page: Page, fitted_server_url: str) -> None:
    """A double click on a legend item neither fades nor hides any trace."""
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    _open_scores(page, f"{fitted_server_url}/scores")
    target = "#scores-trajectories"

    _legend_item(page, target, 1).dblclick()
    page.wait_for_timeout(DOUBLE_CLICK_DELAY_MS)

    assert _opacities(page, target) == [1, 1, 1]
    expect(page.locator(f"{target} .cartesianlayer .trace.scatter")).to_have_count(3)


def test_legend_toggles_the_whole_group(page: Page, fitted_server_url: str) -> None:
    """The traces of one legend group fade and come back together."""
    choose_files(page, fitted_server_url, ["s-00", "s-01", "s-02"])
    # The files have no lot, so the three trajectories share the missing value.
    _open_scores(page, f"{fitted_server_url}/scores?trajectory_color=lot")
    target = "#scores-trajectories"
    expect(page.locator(f"{target} .legend .traces")).to_have_count(1)

    _click_legend(page, target, 0)

    assert _opacities(page, target) == [INACTIVE_OPACITY] * 3

    # The group is inactive, so a right click makes only it active.
    _click_legend(page, target, 0, button="right")

    assert _opacities(page, target) == [1, 1, 1]


def test_inactive_point_click_does_nothing_but_hovers(
    page: Page, fitted_server_url: str
) -> None:
    """A click on a point of an inactive trace shows no row, but its hover label shows."""
    url = f"{fitted_server_url}/scores?color=stem"
    _open_scores(page, url)
    target = "#scores-scatter"
    _click_legend(page, target, 1)
    assert _opacities(page, target)[:3] == [1, INACTIVE_OPACITY, 1]

    # Inactive traces are drawn in a layer of their own, so the point is
    # found from its data instead of the order of the drawn points.
    center = page.evaluate(
        """(target) => {
          const gd = document.querySelector(target);
          const area = gd.querySelector(".nsewdrag").getBoundingClientRect();
          const { xaxis, yaxis } = gd._fullLayout;
          return [
            area.left + xaxis.c2p(gd.calcdata[1][0].x),
            area.top + yaxis.c2p(gd.calcdata[1][0].y),
          ];
        }""",
        target,
    )
    page.mouse.move(*center)
    expect(page.locator(f"{target} .hoverlayer .hovertext")).to_contain_text("s-01")
    page.mouse.click(*center)
    page.wait_for_timeout(DOUBLE_CLICK_DELAY_MS)

    expect(page.locator("#point-table")).to_have_attribute("data-shown-count", "0")
    assert page.url == url


def test_trend_redraw_starts_active_and_isolates_again(
    page: Page, fitted_server_url: str
) -> None:
    """After a redraw of the trends, a right click isolates the trace again."""
    choose_files(page, fitted_server_url, ["s-00", "s-01"])
    page.goto(fitted_server_url + "/explore?view=raw&segment=1:1")
    root = page.locator("[data-explore]")
    expect(root).to_have_attribute("data-trend-step-time", "1")
    target = "#explore-trend-step-time"

    _click_legend(page, target, 0, button="right")
    assert _opacities(page, target) == [1, INACTIVE_OPACITY]

    page.locator("#explore-step-time").press("End")
    expect(root).to_have_attribute("data-trend-step-time", "3")
    assert _opacities(page, target) == [1, 1]

    _click_legend(page, target, 0, button="right")

    assert _opacities(page, target) == [1, INACTIVE_OPACITY]
