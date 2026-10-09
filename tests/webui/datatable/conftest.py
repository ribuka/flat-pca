"""Fixtures of the data table tests: a small frame of every column type."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from flat_pca.webui.datatable import ColumnConfig, TableConfig


@pytest.fixture
def frame() -> pl.DataFrame:
    """Return rows with text, category, integer, float, and datetime columns.

    Every column but ``key`` and ``name`` has a null value.
    """
    return pl.DataFrame(
        {
            "key": ["k1", "k2", "k3", "k4"],
            "name": ["Alpha", "beta", "Gamma", "alphabet"],
            "group": ["a", "b", None, "a"],
            "count": [3, None, 1, 2],
            "score": [0.5, 2.25, None, 1.0],
            "when": [
                datetime.fromisoformat("2026-01-01"),
                datetime.fromisoformat("2026-03-01T12:00"),
                datetime.fromisoformat("2026-02-01"),
                None,
            ],
            "note": ["first", None, "third", "fourth"],
        },
        schema_overrides={"count": pl.Int64},
    )


@pytest.fixture
def config() -> TableConfig:
    """Return a selectable table over every column of ``frame``."""
    return TableConfig(
        table_id="t",
        key="key",
        columns=(
            ColumnConfig("key", label="Key", frame_order=True, title_column="note"),
            ColumnConfig("name", filter="text"),
            ColumnConfig("group", filter="choice"),
            ColumnConfig("count", filter="number"),
            ColumnConfig("score", filter="number"),
            ColumnConfig("when", filter="datetime"),
            ColumnConfig("note", sortable=False),
        ),
        url="/table",
        page_size=2,
        selectable=True,
        selection_name="keys",
        default_sort="key",
    )
