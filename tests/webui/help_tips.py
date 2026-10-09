"""Helpers to read the help tips (``macros/help_tip.html``) of a rendered page."""

from __future__ import annotations

import re


def help_tip_text(html: str, tip_id: str) -> str:
    """Return the explanation of a help tip, checking that its icon refers to it.

    Parameters
    ----------
    html : str
        Rendered page.
    tip_id : str
        ``id`` passed to the ``help_tip`` macro.

    Returns
    -------
    str
        Inner HTML of the explanation, with runs of whitespace collapsed.

    Raises
    ------
    AssertionError
        If the page has no such help tip or its icon does not refer to it.
    """
    assert f'class="help-tip-button" aria-label="Help" aria-describedby="{tip_id}"' in html
    match = re.search(
        rf'<span class="help-tip-text" role="tooltip" id="{re.escape(tip_id)}">(.*?)</span>\s*</span>',
        html,
        re.DOTALL,
    )
    assert match is not None, f"no help tip {tip_id}"
    return " ".join(match.group(1).split())
