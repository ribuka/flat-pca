"""Tests for the data table templates rendered without a web framework."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime

import polars as pl
import pytest
from jinja2 import Environment, select_autoescape

from flat_pca.webui.datatable import (
    ColumnConfig,
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
    assert 'name="t.eq__group" value="a" data-dt-query checked>' in html
    assert 'name="t.eq__group" value="b" data-dt-query>' in html
    assert re.search(r'type="number" id="t-filter-min-\d+" name="t.min__count"', html)
    assert re.search(r'type="datetime-local" id="t-filter-max-\d+" name="t.max__when"', html)
    assert 'data-dt-sort="note"' not in html
    assert html.index('data-dt-key="k4"') < html.index('data-dt-key="k1"')
    assert re.search(r'value="k4"[^>]*data-dt-row-check checked', html)
    assert not re.search(r'value="k1"[^>]*data-dt-row-check checked', html)
    assert '<td class="dt-pinned" title="fourth">k4</td>' in html
    assert "<td>2026-01-01 00:00:00</td>" in html
    assert "1–2 of 2" in html
    assert "data-dt-page=" not in html


def _menu(html: str, column: str) -> str:
    """Return the column menu of ``column``."""
    match = re.search(rf'<div popover id="t-menu-\d+"[^>]*data-dt-menu="{column}">.*?</th>', html, re.DOTALL)
    assert match is not None
    return match[0]


def _header(html: str, column: str) -> str:
    """Return the header cell of ``column`` up to its menu."""
    match = re.search(rf'<th[^>]*>\s*<div class="dt-head">\s*<button[^>]*data-dt-column="{column}">.*?<div popover', html, re.DOTALL)
    assert match is not None
    return match[0]


def test_fragment_opens_a_menu_from_each_column_name(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Each header holds its name, sort mark, and type; the controls are in its menu."""
    state = parse_state({"t.sort": ["count"], "t.order": ["asc"]}, config)
    view = apply_state(frame, state, config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    for index, column in enumerate(config.columns, start=1):
        header = _header(html, column.name)
        assert f'popovertarget="t-menu-{index}"' in header
        assert f'<span class="dt-label">{column.header}</span>' in header
        assert f'<span class="dt-type">{view.types[column.name]}</span>' in header
        assert "data-dt-query" not in header
        menu = _menu(html, column.name)
        assert f'id="t-menu-{index}"' in menu
        assert f'aria-label="{column.header} menu"' in menu
        assert f'data-dt-copy="{column.name}"' in menu
        assert ('data-dt-sort="' in menu) == column.sortable
        assert (f'name="t.null__{column.name}"' in menu) == (column.filter is not None)
    assert '<span class="dt-type">datetime[μs]</span>' in html
    assert '<span class="dt-sort-mark" aria-hidden="true">▲</span>' in _header(html, "count")
    count_menu = _menu(html, "count")
    assert 'data-dt-order="asc" aria-pressed="true"' in count_menu
    assert 'data-dt-order="desc" aria-pressed="false"' in count_menu
    assert 'data-dt-sort="" data-dt-order="asc">Clear sort' in count_menu
    assert 'data-dt-sort="" data-dt-order="asc" disabled>Clear sort' in _menu(html, "score")


def test_fragment_shows_choice_and_null_counts_and_filter_marks(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Menus count values and nulls; filtered columns carry a mark in the header."""
    state = parse_state(
        {"t.eq__group": ["b", "gone"], "t.null__when": ["is_not_null"], "t.q__name": ["a"]}, config
    )
    view = apply_state(frame, state, config, {"group": ["a", "b"]})

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    group = _menu(html, "group")
    values = re.findall(
        r'value="([^"]*)" data-dt-query( checked)?>\s*<span class="dt-choice-value">[^<]*</span> <span class="dt-count">(\d+)</span>',
        group,
    )
    assert values == [("a", "", "2"), ("b", " checked", "1"), ("gone", " checked", "0")]
    assert 'Null values <span class="dt-count">1</span>' in group
    when = _menu(html, "when")
    assert 'name="t.null__when" value="is_not_null" data-dt-query checked>' in when
    assert 'name="t.null__when" value="" data-dt-query>' in when
    assert 'name="t.null__count" value="" data-dt-query checked>' in _menu(html, "count")
    marked = [
        column.name for column in config.columns if "data-dt-filtered" in _header(html, column.name)
    ]
    assert marked == ["name", "group", "when"]


def test_fragment_puts_apply_cancel_and_clear_under_the_filters(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """A menu with filters ends in Apply (disabled until a draft differs), Cancel, and Clear (disabled while unfiltered)."""
    state = parse_state({"t.eq__group": ["b"]}, config)
    view = apply_state(frame, state, config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    for column in config.columns:
        menu = _menu(html, column.name)
        assert ("data-dt-apply disabled>" in menu) == (column.filter is not None)
        assert ("data-dt-cancel>" in menu) == (column.filter is not None)
        assert ("data-dt-draft-status hidden>Unapplied changes" in menu) == (column.filter is not None)
        if column.filter is not None:
            assert menu.index("data-dt-copy=") < menu.index("data-dt-apply")
    assert re.search(r"data-dt-clear-filter>", _menu(html, "group"))
    assert re.search(r"data-dt-clear-filter disabled>", _menu(html, "when"))


def test_fragment_lists_the_rows_of_every_filter_for_the_header_checkbox(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The header checkbox gets every filtered row, beyond the shown page."""
    state = parse_state(
        {"t.eq__group": ["a", "b"], "t.null__count": ["is_not_null"], "t.q__name": ["A"]}, config
    )
    view = apply_state(frame, state, config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    matching = re.search(r"<script type=\"application/json\" data-dt-matching>(.*?)</script>", html)
    assert matching is not None
    assert json.loads(matching[1]) == ["k1", "k4"]


def _footer(html: str) -> str:
    """Return the footer under the table."""
    match = re.search(r'<div class="dt-footer">.*</nav>\s*</div>', html, re.DOTALL)
    assert match is not None
    return match[0]


def test_fragment_moves_between_pages(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Several pages have first, previous, next, last, and a page number input."""
    call = "{{ fragment(config, view) }}"
    first = _render(
        environment, call, config=config, view=apply_state(frame, TableState(sort_by="key"), config)
    )
    last = _render(
        environment,
        call,
        config=config,
        view=apply_state(frame, TableState(sort_by="key", page=2), config),
    )

    assert "1–2 of 4" in first
    assert re.search(r'data-dt-page="1" aria-label="First page"[^>]* disabled>', first)
    assert re.search(r'data-dt-page="0" aria-label="Previous page"[^>]* disabled>', first)
    assert re.search(r'data-dt-page="2" aria-label="Next page"[^>]*">', first)
    assert re.search(r'data-dt-page="2" aria-label="Last page"[^>]*">', first)
    assert re.search(
        r'<input type="number" class="dt-page-input" min="1" max="2" step="1" value="1"\s+'
        r'aria-label="Page number" data-dt-page-input data-dt-page-current="1"> of 2',
        first,
    )
    assert "3–4 of 4" in last
    assert re.search(r'data-dt-page="1" aria-label="First page"[^>]*">', last)
    assert re.search(r'data-dt-page="2" aria-label="Last page"[^>]* disabled>', last)
    assert 'value="2"\n      aria-label="Page number"' in last


def test_fragment_of_one_page_has_no_page_controls(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """One page shows only "a–b of N", and no rows "0 of 0"."""
    single = replace(config, page_size=10)
    every = apply_state(frame, TableState(), single)
    none = apply_state(frame, parse_state({"t.q__name": ["zzz"]}, single), single)
    call = "{{ fragment(config, view) }}"

    one_page = _footer(_render(environment, call, config=single, view=every))
    empty = _footer(_render(environment, call, config=single, view=none))

    assert "1–4 of 4" in one_page
    assert "data-dt-page=" not in one_page
    assert "data-dt-page-input" not in one_page
    assert "0 of 0" in empty
    assert "0 rows, <span data-dt-column-count>7</span> columns" in empty


def test_fragment_footer_shows_the_shape_and_the_selection(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The footer has the matching rows and columns, the count, and Clear."""
    state = parse_state({"t.eq__group": ["a"]}, config)
    view = apply_state(frame, state, config)
    call = "{{ fragment(config, view, selected) }}"

    some = _footer(_render(environment, call, config=config, view=view, selected=["k2", "k3"]))
    none = _footer(_render(environment, call, config=config, view=view, selected=[]))
    limited = replace(config, max_selected=3)
    with_limit = _footer(
        _render(
            environment,
            call,
            config=limited,
            view=apply_state(frame, state, limited),
            selected=["k2", "k3"],
        )
    )

    assert '<span class="dt-shape" data-dt-shape>2 rows, <span data-dt-column-count>7</span> columns</span>' in some
    assert "<span class=\"dt-selected-count\" data-dt-selected-count>2 selected</span>" in some
    assert "<button type=\"button\" data-dt-clear>Clear</button>" in some
    assert "<button type=\"button\" data-dt-clear disabled>Clear</button>" in none
    assert "data-dt-selected-count>0 selected</span>" in none
    assert "data-dt-selected-count>2 / 3 selected</span>" in with_limit


def test_fragment_of_a_plain_table_has_no_selection_state(
    environment: Environment, frame: pl.DataFrame
) -> None:
    """A table without selection shows its shape but no count, and pins its first column."""
    config = TableConfig(
        table_id="p", key="key", columns=(ColumnConfig("key"), ColumnConfig("name")), url="/p"
    )
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert "4 rows, <span data-dt-column-count>2</span> columns" in html
    assert "data-dt-selected-count" not in html
    assert "data-dt-clear" not in html
    assert "dt-check" not in html
    assert '<th class="dt-pinned">' in html
    assert '<td class="dt-pinned">k1</td>' in html
    assert html.count("dt-pinned") == 1 + 4


def test_fragment_pins_the_checkbox_and_first_columns(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The checkbox column and the first column are pinned in the header and every row."""
    view = apply_state(frame, TableState(sort_by="key"), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert '<th class="dt-check dt-pinned"><input type="checkbox" data-dt-check-all' in html
    assert re.search(r'<th class="dt-pinned" aria-sort="ascending">\s*<div class="dt-head">\s*<button[^>]*data-dt-column="key"', html)
    assert html.count('<td class="dt-check dt-pinned">') == 2
    assert html.count('<td class="dt-pinned" title=') == 2
    assert html.count("dt-pinned") == 2 + 2 * 2


def test_fragment_row_checkboxes_use_the_matching_key_text(environment: Environment) -> None:
    """A datetime key has the same text in the row checkbox and the matching keys."""
    frame = pl.DataFrame({"when": [datetime.fromisoformat("2026-01-01")]})
    config = TableConfig(
        table_id="d", key="when", columns=(ColumnConfig("when"),), url="/", selectable=True
    )
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    (key,) = view.matching_keys
    assert f'value="{key}"' in html
    assert f'data-dt-key="{key}"' in html


def _row_check(html: str, key: str) -> str:
    """Return the row checkbox of ``key``."""
    match = re.search(rf'<input type="checkbox" value="{key}"[^>]*>', html)
    assert match is not None
    return match[0]


def _header_check(html: str) -> str:
    """Return the header checkbox."""
    match = re.search(r"<input [^>]*data-dt-check-all[^>]*>", html)
    assert match is not None
    return match[0]


def _limit_notice(html: str) -> str:
    """Return the opening tag of the limit notice."""
    match = re.search(r"<p [^>]*data-dt-limit[^>]*>", html)
    assert match is not None
    return match[0]


def test_container_carries_the_selection_limit(
    environment: Environment, config: TableConfig
) -> None:
    """The container names the limit only when the table has one."""
    unlimited = _render(environment, "{{ container(config) }}", config=config)
    limited = _render(
        environment, "{{ container(config) }}", config=replace(config, max_selected=2)
    )

    assert "data-dt-max-selected" not in unlimited
    assert 'data-dt-max-selected="2"' in limited


def test_fragment_without_limit_shows_no_limit_state(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """A table without a limit has no notice and no disabled checkboxes."""
    view = apply_state(frame, TableState(sort_by="key"), config)

    html = _render(
        environment, "{{ fragment(config, view, selected) }}", config=config, view=view, selected={"k1"}
    )

    assert "data-dt-limit" not in html
    assert "disabled" not in _row_check(html, "k2")
    assert "disabled" not in _header_check(html)


def test_fragment_below_the_limit_allows_selecting(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Below the limit the rows can be checked and the notice is hidden."""
    limited = replace(config, max_selected=2)
    state = parse_state({"t.eq__group": ["b"]}, limited)
    view = apply_state(frame, state, limited)

    html = _render(
        environment, "{{ fragment(config, view, selected) }}", config=limited, view=view, selected=["k1"]
    )

    assert "hidden" in _limit_notice(html)
    assert "The limit of 2 selected rows is reached" in html
    assert "disabled" not in _row_check(html, "k2")
    assert "disabled" not in _header_check(html)


def test_fragment_at_the_limit_disables_unselected_rows(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """At the limit unselected rows are disabled, selected ones not, and the notice shows."""
    limited = replace(config, max_selected=2)
    view = apply_state(frame, TableState(sort_by="key"), limited)

    html = _render(
        environment,
        "{{ fragment(config, view, selected) }}",
        config=limited,
        view=view,
        selected=["k1", "k3"],
    )

    assert "hidden" not in _limit_notice(html)
    assert "checked" in _row_check(html, "k1")
    assert "disabled" not in _row_check(html, "k1")
    assert "disabled" in _row_check(html, "k2")
    assert "disabled" in _header_check(html)


def test_fragment_header_checkbox_follows_the_limit(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The header is disabled only when selecting every matching row passes the limit."""
    limited = replace(config, max_selected=3)
    every = apply_state(frame, TableState(sort_by="key"), limited)
    group_a = apply_state(frame, parse_state({"t.eq__group": ["a"]}, limited), limited)
    call = "{{ fragment(config, view, selected) }}"

    over = _render(environment, call, config=limited, view=every, selected=["k1"])
    fits = _render(environment, call, config=limited, view=group_a, selected=["k2"])
    all_selected = _render(
        environment, call, config=limited, view=group_a, selected=["k1", "k2", "k3", "k4"]
    )

    assert "disabled" in _header_check(over)
    assert "disabled" not in _header_check(fits)
    assert "disabled" not in _header_check(all_selected)


def test_fragment_counts_keys_differing_only_in_case(environment: Environment) -> None:
    """Keys differing only in case count as separate rows against the limit."""
    frame = pl.DataFrame({"k": ["A", "a", "z"]})
    config = TableConfig(
        table_id="c", key="k", columns=(ColumnConfig("k"),), url="/", selectable=True, max_selected=2
    )
    view = apply_state(frame, TableState(), config)
    call = "{{ fragment(config, view, selected) }}"

    full = _render(environment, call, config=config, view=view, selected=["A", "a"])
    single = replace(config, max_selected=1)
    pair = apply_state(frame.head(2), TableState(), single)
    one = _render(environment, call, config=single, view=pair, selected=[])

    assert "hidden" not in _limit_notice(full)
    assert "disabled" in _row_check(full, "z")
    assert "disabled" in _header_check(one)


def test_fragment_without_toolbar_settings_has_no_toolbar(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """By default the table has no search box, "Columns" menu, or chips."""
    view = apply_state(frame, parse_state({"t.eq__group": ["a"]}, config), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert "dt-toolbar" not in html
    assert "dt-chips" not in html
    assert "data-dt-columns-menu" not in html


def test_fragment_shows_the_search_box_with_the_search(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The search box of the whole table is a query control kept across reloads."""
    searchable = replace(config, search=True)
    view = apply_state(frame, parse_state({"t.search": ["alp"]}, searchable), searchable)

    html = _render(environment, "{{ fragment(config, view) }}", config=searchable, view=view)

    search = re.search(r'<input type="search" id="t-search"[^>]*>', html)
    assert search is not None
    assert 'name="t.search" value="alp"' in search[0]
    assert 'aria-label="Search"' in search[0]
    assert "data-dt-query hx-preserve" in search[0]
    assert html.index('id="t-search"') < html.index('<table class="dt-table">')


def test_fragment_lists_the_filters_in_use_as_chips(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Each chip has a button naming the parameters it clears; no filter, no chips."""
    chipped = replace(config, filter_chips=True)
    state = parse_state({"t.eq__group": ["a", "b"], "t.min__score": ["1"]}, chipped)
    view = apply_state(frame, state, chipped)
    call = "{{ fragment(config, view) }}"

    html = _render(environment, call, config=chipped, view=view)
    unfiltered = _render(
        environment, call, config=chipped, view=apply_state(frame, TableState(), chipped)
    )

    chips = re.findall(r'<li class="dt-chip">.*?</li>', html, re.DOTALL)
    assert len(chips) == 2
    assert '<span class="dt-chip-label">group ∈ {a, b}</span>' in chips[0]
    assert 'aria-label="Remove filter group ∈ {a, b}"' in chips[0]
    assert """data-dt-remove-filter='["t.eq__group"]'""" in chips[0]
    assert '<span class="dt-chip-label">score ≥ 1</span>' in chips[1]
    assert """data-dt-remove-filter='["t.min__score", "t.max__score"]'""" in chips[1]
    assert "dt-chips" not in unfiltered


def test_fragment_has_a_columns_menu_of_every_column(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The "Columns" menu lists every column; the first one cannot be hidden."""
    chooser = replace(config, column_chooser=True)
    view = apply_state(frame, TableState(), chooser)

    html = _render(environment, "{{ fragment(config, view) }}", config=chooser, view=view)

    assert 'popovertarget="t-columns-menu"' in html
    menu = re.search(r'<div popover id="t-columns-menu".*?</div>', html, re.DOTALL)
    assert menu is not None
    assert "data-dt-menu data-dt-columns-menu" in menu[0]
    boxes = re.findall(r'<input type="checkbox" checked data-dt-show-column="([^"]*)"( disabled)?>', menu[0])
    assert boxes == [(column.name, " disabled" if index == 0 else "") for index, column in enumerate(config.columns)]
    assert '<span class="dt-choice-value">Key</span>' in menu[0]


def _histogram(html: str, column: str) -> str:
    """Return the histogram (or top values) under the header of ``column``."""
    header = _header(html, column)
    match = re.search(
        r'<(div class="dt-histogram"|ul class="dt-top-values").*?</(div>\s*</div|ul)>', header, re.DOTALL
    )
    assert match is not None
    return match[0]


def test_fragment_draws_histograms_under_the_column_types(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Number and datetime headers get an SVG histogram; bins outside the bounds fade."""
    config = replace(config, histograms=True)
    state = parse_state({"t.min__count": ["2"]}, config)
    view = apply_state(frame, state, config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    count = _histogram(html, "count")
    assert 'viewBox="0 0 30 32"' in count
    assert 'aria-label="Histogram of count, 1 to 3"' in count
    assert re.findall(r'<g class="dt-bin( dt-out)?"><title>([^<]*)</title>', count) == [
        (" dt-out", "1: 1 row"),
        ("", "2: 1 row"),
        ("", "3: 1 row"),
    ]
    assert '<span>1</span><span>3</span>' in count
    assert count.count('y="0.00" width="8" height="32.00"></rect></g>') == 3
    when = _histogram(html, "when")
    assert when.count('<g class="dt-bin') == 20
    assert "<title>2026-01-01 00:00:00 – 2026-01-03 23:24:00: 1 row</title>" in when
    assert "0 rows</title>" in when
    assert '<span>2026-01-01 00:00:00</span><span>2026-03-01 12:00:00</span>' in when
    for column in ("key", "name", "group", "note"):
        assert "dt-histogram" not in _header(html, column)


def test_fragment_shows_the_top_values_of_category_columns(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """A categorical header lists its values with bars; unchecked values fade."""
    config = replace(config, histograms=True)
    frame = frame.with_columns(pl.col("group").cast(pl.Categorical))
    state = parse_state({"t.eq__group": ["b"]}, config)
    view = apply_state(frame, state, config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    group = _histogram(html, "group")
    assert 'aria-label="Most frequent values of group"' in group
    assert re.findall(
        r'<li class="dt-top-value( dt-out)?" title="([^"]*)">\s*<span class="dt-top-bar" style="width: ([\d.]+)%"></span>\s*'
        r'<span class="dt-top-label">([^<]*)</span> <span class="dt-count">(\d+)</span>',
        group,
    ) == [(" dt-out", "a: 2 rows", "100.0", "a", "2"), ("", "b: 1 row", "50.0", "b", "1")]
    assert "dt-top-others" not in group


def test_fragment_counts_the_other_values_of_category_columns(environment: Environment) -> None:
    """Values past the five most frequent are counted together as Others."""
    config = TableConfig(
        table_id="t", key="key", columns=(ColumnConfig("lot"),), url="/t", histograms=True
    )
    lots = ["a", "a", "b", "c", "d", "e", "f", "g"]
    frame = pl.DataFrame(
        {"key": [str(i) for i in range(len(lots))], "lot": lots},
        schema_overrides={"lot": pl.Categorical},
    )
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    group = _histogram(html, "lot")
    assert group.count('class="dt-top-bar"') == 5
    assert re.search(
        r'<li class="dt-top-value dt-top-others" title="Other values: 2 rows">\s*'
        r'<span class="dt-top-label">Others</span> <span class="dt-count">2</span>',
        group,
    )


def test_fragment_without_histograms_draws_none(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Histograms are off by default."""
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert "dt-histogram" not in html
    assert "dt-top-values" not in html


def _export_menu(html: str) -> str:
    """Return the "Export" menu of table ``t``."""
    menu = re.search(r'<div popover id="t-export-menu".*?data-dt-export-error hidden></p>\s*</div>', html, re.DOTALL)
    assert menu is not None
    return menu[0]


def test_container_carries_the_export_url(environment: Environment, config: TableConfig) -> None:
    """A table with export names the URL that answers it; one without has none."""
    exported = _render(
        environment, "{{ container(config) }}", config=replace(config, export_url="/table/export")
    )

    container = re.search(r'<div id="t"[^>]*>', exported)
    assert container is not None
    assert 'data-dt-export-url="/table/export"' in container[0]
    assert "data-dt-export-url" not in _render(environment, "{{ container(config) }}", config=config)


def test_fragment_has_an_export_menu_of_the_filtered_and_selected_rows(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The "Export" menu offers CSV and Parquet for the filtered and the selected rows."""
    exporting = replace(config, export_url="/table/export")
    view = apply_state(frame, parse_state({"t.eq__group": ["a"]}, exporting), exporting)

    html = _render(
        environment, "{{ fragment(config, view, selected) }}", config=exporting, view=view, selected={"k2", "k3", "k4"}
    )

    assert 'class="dt-toolbar"' in html
    assert 'popovertarget="t-export-menu"' in html
    menu = _export_menu(html)
    assert "data-dt-menu data-dt-export-menu" in menu
    assert re.findall(r'data-dt-export="([^"]*)" data-dt-export-rows="([^"]*)"', menu) == [
        ("csv", "filtered"), ("parquet", "filtered"), ("csv", "selected"), ("parquet", "selected")
    ]
    assert 'Filtered rows <span class="dt-count">2</span>' in menu
    assert 'Selected rows <span class="dt-count" data-dt-export-selected-count>3</span>' in menu
    assert 'aria-label="Export selected rows as Parquet">Parquet</button>' in menu
    assert " disabled" not in menu


def test_fragment_export_menu_disables_rows_without_any(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Without a matching or a selected row, its buttons are disabled."""
    exporting = replace(config, export_url="/table/export")
    view = apply_state(frame, TableState(), exporting)
    empty = apply_state(frame, parse_state({"t.eq__group": ["z"]}, exporting), exporting)

    menu = _export_menu(_render(environment, "{{ fragment(config, view) }}", config=exporting, view=view))
    none_matching = _export_menu(
        _render(environment, "{{ fragment(config, view, ['k1']) }}", config=exporting, view=empty)
    )

    assert re.findall(r'data-dt-export-rows="(\w+)"[^>]*?( disabled)?>', menu) == [
        ("filtered", ""), ("filtered", ""), ("selected", " disabled"), ("selected", " disabled")
    ]
    assert re.findall(r'data-dt-export-rows="(\w+)"[^>]*?( disabled)?>', none_matching) == [
        ("filtered", " disabled"), ("filtered", " disabled"), ("selected", ""), ("selected", "")
    ]


def test_fragment_of_a_plain_table_exports_only_the_filtered_rows(
    environment: Environment, frame: pl.DataFrame
) -> None:
    """A table without selection has no selected rows to export."""
    plain = TableConfig(
        table_id="t", key="key", columns=(ColumnConfig("key"),), url="/t", export_url="/t/export"
    )
    view = apply_state(frame, TableState(), plain)

    menu = _export_menu(_render(environment, "{{ fragment(config, view) }}", config=plain, view=view))

    assert re.findall(r'data-dt-export-rows="([^"]*)"', menu) == ["filtered", "filtered"]


def test_fragment_without_export_has_no_export_menu(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Export is off by default, for example in a dialog of chosen rows."""
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    assert "Export" not in html
    assert "data-dt-export" not in html


def test_fragment_has_a_pin_toggle_off_by_default(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The "Pin columns" toggle is a button, not pressed until the browser shows the choice."""
    pinnable = replace(config, pin_toggle=True)
    view = apply_state(frame, TableState(), pinnable)
    call = "{{ fragment(config, view) }}"

    html = _render(environment, call, config=pinnable, view=view)
    plain = _render(environment, call, config=config, view=view)

    toggle = re.search(r"<button[^>]*data-dt-pin-toggle[^>]*>.*?</button>", html)
    assert toggle is not None
    assert 'type="button"' in toggle[0]
    assert 'aria-pressed="false"' in toggle[0]
    assert toggle[0].endswith("Pin columns</button>")
    assert html.index("dt-toolbar") < html.index("data-dt-pin-toggle")
    assert "data-dt-pin-toggle" not in plain


def test_fragment_has_a_resize_handle_on_every_column_header(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """Each column header but the checkbox column's ends in a focusable handle naming its column."""
    view = apply_state(frame, TableState(), config)

    html = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    handles = re.findall(r'<span class="dt-resizer"[^>]*></span>\s*</th>', html)
    assert len(handles) == len(config.columns)
    for handle, column in zip(handles, config.columns, strict=True):
        assert 'role="separator" aria-orientation="vertical"' in handle
        assert f'aria-label="Resize column {column.header}"' in handle
        assert 'tabindex="0"' in handle
        assert f'data-dt-resize="{column.name}"' in handle
    check = re.search(r'<th class="dt-check dt-pinned">.*?</th>', html, re.DOTALL)
    assert check is not None
    assert "dt-resizer" not in check[0]


def test_columns_menu_resets_the_column_widths(
    environment: Environment, config: TableConfig, frame: pl.DataFrame
) -> None:
    """The "Columns" menu ends with "Reset column widths", disabled until the browser enables it."""
    chooser = replace(config, column_chooser=True)
    view = apply_state(frame, TableState(), chooser)

    html = _render(environment, "{{ fragment(config, view) }}", config=chooser, view=view)
    plain = _render(environment, "{{ fragment(config, view) }}", config=config, view=view)

    menu = re.search(r'<div popover id="t-columns-menu".*?</fieldset>.*?</div>', html, re.DOTALL)
    assert menu is not None
    assert '<button type="button" data-dt-reset-widths disabled>Reset column widths</button>' in menu[0]
    assert "data-dt-reset-widths" not in plain
