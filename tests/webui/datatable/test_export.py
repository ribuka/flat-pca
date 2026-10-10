"""Tests for exporting the rows of a data table as a CSV or Parquet file."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import unquote

import polars as pl
import pytest

from flat_pca.webui.datatable import (
    ColumnConfig,
    ExportFile,
    ExportRequest,
    TableConfig,
    TableState,
    apply_state,
    export_file,
    export_filename,
    export_rows,
    file_content,
    parse_export,
    parse_selection,
    parse_state,
)

MOMENT = datetime(2026, 10, 10, 9, 5, 7, tzinfo=UTC)


def _fields(rows: str = "filtered", file_format: str = "csv", **more: list[str]) -> dict[str, list[str]]:
    """Return the fields of an export of table ``t``, plus ``more``."""
    return {"t.export_format": [file_format], "t.export_rows": [rows], **more}


def _keys(rows: pl.DataFrame) -> list[str]:
    """Return the keys of exported rows."""
    return rows["key"].to_list()


def test_parse_export_reads_the_format_rows_state_and_selection(config: TableConfig) -> None:
    """The table's own parameters give the state; the page is ignored."""
    request = parse_export(
        _fields(
            "selected",
            "parquet",
            **{
                "t.sort": ["score"],
                "t.order": ["desc"],
                "t.page": ["2"],
                "t.eq__group": ["a"],
                "keys": ['["k2", "k1"]'],
                "other": ["ignored"],
            },
        ),
        config,
    )

    assert request == ExportRequest(
        file_format="parquet",
        rows="selected",
        state=TableState(sort_by="score", descending=True, page=2, equals={"group": ("a",)}),
        selected=("k2", "k1"),
    )


def test_parse_export_of_the_filtered_rows_ignores_the_selection(config: TableConfig) -> None:
    """Only an export of the selected rows reads the selection."""
    request = parse_export(_fields(keys=["not json"]), config)

    assert (request.rows, request.selected) == ("filtered", ())


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"t.export_rows": ["filtered"]}, "t.export_format"),
        (_fields(file_format="xlsx"), "t.export_format"),
        (_fields(rows="page"), "t.export_rows"),
        (_fields(**{"t.export_columns": ["all"]}), "unknown parameter"),
        (_fields(**{"t.min__group": ["1"]}), "does not fit"),
        (_fields("selected", keys=["{}"]), "JSON array of strings"),
        (_fields("selected", keys=["[1]"]), "JSON array of strings"),
        (_fields("selected", keys=["["]), "JSON array"),
    ],
)
def test_parse_export_rejects_invalid_fields(
    config: TableConfig, fields: dict[str, list[str]], message: str
) -> None:
    """A missing or unknown format or rows, a bad filter, or a bad selection is a ValueError."""
    with pytest.raises(ValueError, match=message):
        parse_export(fields, config)


def test_parse_export_needs_a_selectable_table_for_selected_rows(config: TableConfig) -> None:
    """A table without selection has no selected rows to export."""
    with pytest.raises(ValueError, match="without selection"):
        parse_export(_fields("selected"), replace(config, selectable=False))


def test_parse_export_without_a_selection_exports_no_row(config: TableConfig) -> None:
    """A missing selection field is an empty selection."""
    assert parse_export(_fields("selected"), config).selected == ()


def test_parse_selection_reads_a_json_array() -> None:
    """The selection is one JSON array of strings."""
    assert parse_selection('["a", "b"]') == ("a", "b")


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"t.sort": ["score"]},
        {"t.sort": ["score"], "t.order": ["desc"]},
        {"t.sort": ["when"], "t.order": ["desc"]},
        {"t.sort": ["key"], "t.order": ["desc"]},
        {"t.q__name": ["ALPHA"]},
        {"t.eq__group": ["a", "b"]},
        {"t.min__count": ["2"]},
        {"t.max__score": ["1.5"], "t.sort": ["score"]},
        {"t.min__when": ["2026-02-01T00:00"]},
        {"t.null__group": ["is_null"]},
        {"t.null__count": ["is_not_null"], "t.order": ["desc"]},
        {"t.search": ["a"], "t.sort": ["name"]},
        {"t.eq__group": ["a"], "t.null__score": ["is_not_null"], "t.sort": ["count"]},
    ],
)
def test_filtered_export_matches_the_table_over_every_page(
    config: TableConfig, frame: pl.DataFrame, parameters: dict[str, list[str]]
) -> None:
    """The filtered rows are the table's matching rows, on all its pages, in its order."""
    searchable = replace(config, search=True)
    state = parse_state(parameters, searchable)
    view = apply_state(frame, state, searchable)

    rows = export_rows(frame, ExportRequest("csv", "filtered", state), searchable)

    assert _keys(rows) == view.matching_keys
    assert rows.height == view.page.total


def test_filtered_export_takes_every_page(config: TableConfig, frame: pl.DataFrame) -> None:
    """Rows on pages other than the shown one are exported too."""
    state = TableState(sort_by="key", page=2)

    rows = export_rows(frame, ExportRequest("csv", "filtered", state), config)

    assert config.page_size < rows.height == frame.height


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (TableState(sort_by="key"), ["k1", "k2", "k4"]),
        (TableState(sort_by="key", descending=True), ["k4", "k2", "k1"]),
        (TableState(sort_by="score"), ["k1", "k4", "k2"]),
        (TableState(sort_by="count", descending=True), ["k1", "k4", "k2"]),
        (TableState(sort_by="key", equals={"group": ("b",)}), ["k1", "k2", "k4"]),
        (TableState(sort_by="key", search="nothing"), ["k1", "k2", "k4"]),
    ],
)
def test_selected_export_keeps_every_selected_row_in_the_sort_order(
    config: TableConfig, frame: pl.DataFrame, state: TableState, expected: list[str]
) -> None:
    """Selected rows are exported also when the filters hide them, in the table's order."""
    request = ExportRequest("csv", "selected", state, ("k4", "k1", "k2", "missing"))

    assert _keys(export_rows(frame, request, config)) == expected


def test_selected_export_matches_keys_as_text(frame: pl.DataFrame) -> None:
    """Keys of another type are selected by their text, as the checkboxes hold them."""
    numbered = frame.with_columns(pl.Series("key", [10, 20, 30, 40]))
    config = TableConfig(
        table_id="t", key="key", columns=(ColumnConfig("key"),), url="/t", selectable=True
    )

    rows = export_rows(numbered, ExportRequest("csv", "selected", TableState(), ("30", "10")), config)

    assert rows["key"].to_list() == [10, 30]


def test_export_writes_the_shown_columns_in_display_order(frame: pl.DataFrame) -> None:
    """Columns not shown in the table, such as a tooltip column, are left out."""
    config = TableConfig(
        table_id="t",
        key="key",
        columns=(ColumnConfig("score"), ColumnConfig("key", title_column="note"), ColumnConfig("when")),
        url="/t",
    )

    rows = export_rows(frame.lazy(), ExportRequest("csv", "filtered", TableState()), config)

    assert rows.columns == ["score", "key", "when"]


def test_csv_is_utf8_with_a_bom_and_iso_datetimes(frame: pl.DataFrame) -> None:
    """Excel reads the BOM as UTF-8; datetimes are ISO 8601; nulls are empty."""
    rows = frame.select("key", "name", "when").with_columns(
        pl.Series("name", ["Älpha", "β", "γ", "δ"])
    )

    content = file_content(rows, "csv")

    assert content.startswith(b"\xef\xbb\xbf")
    lines = content[3:].decode("utf-8").splitlines()
    assert lines[0] == "key,name,when"
    assert lines[1].startswith("k1,Älpha,")
    written = [line.split(",")[2] for line in lines[1:]]
    assert [datetime.fromisoformat(text) if text else None for text in written] == frame["when"].to_list()
    assert written[1].startswith("2026-03-01T12:00:00")


def test_parquet_keeps_the_rows_and_column_types(frame: pl.DataFrame) -> None:
    """The Parquet file reads back as the same frame."""
    rows = frame.with_columns(pl.col("group").cast(pl.Categorical))

    content = file_content(rows, "parquet")

    assert pl.read_parquet(io.BytesIO(content)).equals(rows)


def test_export_filename_has_the_name_and_time(config: TableConfig) -> None:
    """Files are named ``<export_name>_<YYYYmmdd-HHMMSS>.<format>``."""
    named = replace(config, export_name="catalog")

    assert export_filename(named, "csv", MOMENT) == "catalog_20261010-090507.csv"
    assert export_filename(config, "parquet", MOMENT) == "table_20261010-090507.parquet"


def test_export_file_headers_download_the_file() -> None:
    """The file downloads as an attachment, also under a name beyond ASCII."""
    plain = ExportFile(b"", "catalog_20261010-090507.csv", "text/csv; charset=utf-8")
    named = ExportFile(b"", 'données "x".csv', "text/csv; charset=utf-8")

    assert plain.headers == {
        "Content-Disposition": (
            "attachment; filename=\"catalog_20261010-090507.csv\"; "
            "filename*=UTF-8''catalog_20261010-090507.csv"
        )
    }
    disposition = named.headers["Content-Disposition"]
    assert 'filename="donn?es _x_.csv"' in disposition
    assert unquote(disposition.split("filename*=UTF-8''")[1]) == 'données "x".csv'


def test_export_file_writes_the_requested_rows(config: TableConfig, frame: pl.DataFrame) -> None:
    """A filtered CSV export from parsed fields holds the matching rows in order."""
    request = parse_export(
        _fields(**{"t.eq__group": ["a"], "t.sort": ["count"], "t.order": ["desc"]}), config
    )

    file = export_file(frame, request, config, MOMENT)

    assert file.filename == "table_20261010-090507.csv"
    assert file.media_type == "text/csv; charset=utf-8"
    exported = pl.read_csv(io.BytesIO(file.content[3:]))
    assert exported.columns == [column.name for column in config.columns]
    assert exported["key"].to_list() == ["k1", "k4"]


def test_export_file_writes_the_selected_rows_as_parquet(
    config: TableConfig, frame: pl.DataFrame
) -> None:
    """A selected Parquet export holds the selected rows in the sort order."""
    request = parse_export(
        _fields("selected", "parquet", **{"t.sort": ["score"], "keys": [json.dumps(["k2", "k3"])]}),
        config,
    )

    file = export_file(frame, request, config, MOMENT)

    assert file.filename.endswith(".parquet")
    assert pl.read_parquet(io.BytesIO(file.content))["key"].to_list() == ["k2", "k3"]
