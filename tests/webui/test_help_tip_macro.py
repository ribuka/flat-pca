"""Tests for the ``help_tip`` Jinja macro."""

from __future__ import annotations

from help_tips import help_tip_text

from flat_pca.webui.templating import templates


def _render(call: str) -> str:
    """Render one ``help_tip`` macro call."""
    source = '{% from "macros/help_tip.html" import help_tip %}' + call
    return templates.env.from_string(source).render()


def test_help_tip_is_a_focusable_icon_described_by_its_explanation() -> None:
    """The icon is a button that names the explanation with aria-describedby."""
    html = _render('{% call help_tip("x-help") %}Explains <sub>x</sub>.{% endcall %}')

    assert html.startswith('<span class="help-tip">')
    assert (
        '<button type="button" class="help-tip-button" aria-label="Help" aria-describedby="x-help">'
        '<span class="material-symbols-outlined" aria-hidden="true">help</span></button>'
    ) in html
    assert '<span class="help-tip-text" role="tooltip" id="x-help">' in html
    assert help_tip_text(html, "x-help") == "Explains <sub>x</sub>."


def test_help_tip_label_names_the_icon() -> None:
    """``label`` replaces the accessible name of the icon."""
    html = _render('{% call help_tip("y-help", label="About y") %}y{% endcall %}')

    assert 'aria-label="About y" aria-describedby="y-help"' in html
