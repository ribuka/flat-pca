"""Tests for the data table templates rendered without a web framework."""

from __future__ import annotations

import re

import polars as pl
import pytest
from jinja2 import Environment, select_autoescape

from flat_pca.webui.datatable import (
    TableConfig,
    TableState,
    apply_state,
    configure_environment,
    parse_state,
)


@pytest.fixture
def environment() -> Environment:
    """Return a bare Jinja2 environment with the data table templates."""
    environment = Environment(autoescape=select_autoescape(["html"]))
    configure_environment(environment)
    return environment


def _render(environment: Environment, call: str, **context: object) -> str:
    """Render one macro call of ``datatable/macros.html``."""
    template = environment.from_string(
        '{% from "datatable/macros.html" import container, fragment %}' + call
    )
    return template.render(**context)


def test_container_loads_the_fragment_and_keeps_the_selection(
    environment: Environment, config: TableConfig
) -> None:
    """The container reloads itself; the selection input sits outside it."""
    html = _render(
        environment,
        '{{ container(config, ["k1"], locked=true, triggers="data-changed from:body") }}',
        config=config,
    )

    container = re.search(r'<div id="t"[^>]*>', html)
    assert container is not None
    assert 'hx-get="/table"' in container[0]
    assert 'hx-trigger="load, dt-reload, data-changed from:body"' in container[0]
    assert 'data-dt-selection="t-selection"' in container[0]
    assert "data-dt-locked" in container[0]
    assert """<input type="hidden" id="t-selection" name="keys" value='["k1"]'>""" in html
    assert html.index("</div>") < html.index('id="t-selection"')


def test_container_of_a_plain_table_has_no_selection(environment: Environment) -> None:
    """A table without selection has no selection input."""
    config = TableConfig(table_id="p", key="k", columns=(), url="/p")

    html = _render(environment, "{{ container(config) }}", config=config)

    assert "data-dt-selection" not in html
    assert 'type="hidden"' not in html


def test_fragment_shows_controls_rows_and_pages(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The fragment names its controls by the table and marks the sort and selection."""
    state = parse_state({"t.sort": ["score"], "t.order": ["desc"], "t.eq__group": ["a"]}, config)
    view = apply_state(frame, state, config)

    html = _render(
        environment, "{{ fragment(config, view, selected) }}", config=config, view=view, selected={"k4"}
    )

    assert '<input type="hidden" name="t.sort" value="score" data-dt-query>' in html
    assert '<input type="hidden" name="t.order" value="desc" data-dt-query>' in html
    assert '<input type="hidden" name="t.page" value="1" data-dt-query>' in html
    assert '["k4", "k1"]</script>' in html
    assert html.count('aria-sort="descending"') == 1
    assert 'aria-label="Select all filtered rows"' in html
    assert 'name="t.q__name"' in html
    assert '<option value="a" selected>' in html
    assert re.search(r'type="number" id="t-filter-min-\d+" name="t.min__count"', html)
    assert re.search(r'type="datetime-local" id="t-filter-max-\d+" name="t.max__when"', html)
    assert 'data-dt-sort="note"' not in html
    assert html.index('data-dt-key="k4"') < html.index('data-dt-key="k1"')
    assert re.search(r'value="k4"[^>]*data-dt-row-check checked', html)
    assert not re.search(r'value="k1"[^>]*data-dt-row-check checked', html)
    assert '<td title="fourth">k4</td>' in html
    assert "<td>2026-01-01 00:00:00</td>" in html
    assert "1–2 of 2" in html
    assert "data-dt-page=" not in html


def test_fragment_links_pages(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Several pages show links to them and mark the current one."""
    view = apply_state(frame, TableState(sort_by="key", page=2), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert "3–4 of 4" in html
    assert '<button type="button" data-dt-page="1" >Previous</button>' in html
    assert 'aria-current="page" disabled>2</button>' in html
    assert '<button type="button" data-dt-page="3" disabled>Next</button>' in html
