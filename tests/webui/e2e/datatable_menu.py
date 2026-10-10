"""Opening and closing the column menus of a data table in browser tests."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect


def open_column_menu(table: Locator, header: str) -> Locator:
    """Open the menu of a column by clicking its name.

    Parameters
    ----------
    table : Locator
        Data table container (``[data-datatable]``), or an element holding it.
    header : str
        Header text of the column.

    Returns
    -------
    Locator
        The open menu.
    """
    table.get_by_role("button", name=header, exact=True).click()
    menu = column_menu(table, header)
    expect(menu).to_be_visible()
    return menu


def column_menu(table: Locator, header: str) -> Locator:
    """Return the menu of a column, open or not.

    Parameters
    ----------
    table : Locator
        Data table container, or an element holding it.
    header : str
        Header text of the column.

    Returns
    -------
    Locator
        The menu (``[data-dt-menu]``) named ``"<header> menu"``.
    """
    return table.locator(f'[data-dt-menu][aria-label="{header} menu"]')


def apply_column_menu(menu: Locator) -> None:
    """Apply the draft filters of an open column menu, which closes it.

    Parameters
    ----------
    menu : Locator
        The open menu, with filters changed from those applied.
    """
    menu.get_by_role("button", name="Apply").click()
    expect(menu).to_be_hidden()


def close_column_menu(page: Page, menu: Locator) -> None:
    """Close an open column menu with Escape.

    Parameters
    ----------
    page : Page
        Browser page.
    menu : Locator
        The open menu.
    """
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
