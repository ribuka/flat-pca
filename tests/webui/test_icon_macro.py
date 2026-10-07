"""Tests for the ``icon`` Jinja macro."""

from __future__ import annotations

from flat_pca.webui.templating import templates


def _render(call: str) -> str:
    """Render one ``icon`` macro call."""
    source = '{% from "macros/icon.html" import icon %}' + call
    return templates.env.from_string(source).render()


def test_icon_is_decorative_without_label() -> None:
    """An icon without a label is hidden from assistive technology."""
    assert _render('{{ icon("check_circle") }}') == (
        '<span class="material-symbols-outlined" aria-hidden="true">'
        "check_circle</span>"
    )


def test_icon_with_class_and_label() -> None:
    """Extra classes are appended and a label makes the icon an image."""
    html = _render('{{ icon("check_circle", class="icon-success", label="完了") }}')

    assert html == (
        '<span class="material-symbols-outlined icon-success"'
        ' role="img" aria-label="完了" title="完了">check_circle</span>'
    )
