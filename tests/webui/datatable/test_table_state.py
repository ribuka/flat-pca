"""Tests for parsing a data table's state from query parameters."""

from __future__ import annotations

from datetime import datetime

import pytest

from flat_pca.webui.datatable import ColumnConfig, TableConfig, TableState, parse_state


def test_parse_state_reads_filters_sort_and_page(config: TableConfig) -> None:
    """Every control of the table is parsed; blank values are ignored."""
    state = parse_state(
        {
            "t.q__name": [" alp "],
            "t.eq__group": ["a"],
            "t.min__count": ["2"],
            "t.max__count": [""],
            "t.max__when": ["2026-02-01T00:00"],
            "t.sort": ["score"],
            "t.order": ["desc"],
            "t.page": ["3"],
        },
        config,
    )

    assert state == TableState(
        sort_by="score",
        descending=True,
        page=3,
        text={"name": "alp"},
        equals={"group": "a"},
        ranges={"count": (2.0, None), "when": (None, datetime.fromisoformat("2026-02-01"))},
    )


def test_parse_state_defaults_and_ignores_other_tables(config: TableConfig) -> None:
    """Without parameters the default sort applies; other prefixes are ignored."""
    state = parse_state({"other.sort": ["x"], "q": ["y"], "t.eq__group": ["", "b"]}, config)

    assert state == TableState(sort_by="key", equals={"group": "b"})


def test_parse_state_without_default_sort_keeps_frame_order() -> None:
    """A table without a default sort has no sort column."""
    config = TableConfig(table_id="t", key="k", columns=(ColumnConfig("k"),), url="/")

    assert parse_state({}, config).sort_by is None


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"t.eq__unknown": ["x"]}, "unknown filter column"),
        ({"t.eq__note": ["x"]}, "unknown filter column"),
        ({"t.min__group": ["1"]}, "does not fit"),
        ({"t.eq__count": ["1"]}, "does not fit"),
        ({"t.q__group": ["a"]}, "does not fit"),
        ({"t.min__count": ["abc"]}, "invalid bound"),
        ({"t.max__when": ["yesterday"]}, "invalid bound"),
        ({"t.other": ["1"]}, "unknown parameter"),
        ({"t.sort": ["missing"]}, "unknown sort column"),
        ({"t.sort": ["note"]}, "unknown sort column"),
        ({"t.order": ["up"]}, "invalid sort order"),
        ({"t.page": ["0"]}, "invalid page"),
        ({"t.page": ["x"]}, "invalid page"),
    ],
)
def test_parse_state_rejects_invalid_parameters(
    config: TableConfig, parameters: dict[str, list[str]], message: str
) -> None:
    """Unknown controls and columns, mismatched filters, and bad values are rejected."""
    with pytest.raises(ValueError, match=message):
        parse_state(parameters, config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_size": 0}, "page_size"),
        ({"columns": (ColumnConfig("k"), ColumnConfig("k"))}, "unique"),
        ({"default_sort": "missing"}, "default_sort"),
        ({"selectable": True, "max_selected": 0}, "max_selected must be positive"),
        ({"selectable": True, "max_selected": 1.5}, "max_selected must be an integer"),
        ({"selectable": True, "max_selected": True}, "max_selected must be an integer"),
        ({"max_selected": 2}, "selectable"),
    ],
)
def test_table_config_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    """The page size, column names, default sort, and selection limit are checked."""
    settings: dict[str, object] = {
        "table_id": "t",
        "key": "k",
        "columns": (ColumnConfig("k"),),
        "url": "/",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        TableConfig(**settings)  # type: ignore[arg-type]
