"""Exporting the rows of a data table as a CSV or Parquet file."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args
from urllib.parse import quote

import polars as pl

from .config import TableConfig
from .query import key_text, matching_rows
from .state import TableState, parse_state

EXPORT_FORMAT_PARAMETER = "export_format"
EXPORT_ROWS_PARAMETER = "export_rows"

# The file formats of an export and the rows it holds: every row matching
# the filters (on all pages), or the selected rows.
type ExportFormat = Literal["csv", "parquet"]
type ExportRows = Literal["filtered", "selected"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = get_args(ExportFormat.__value__)
EXPORT_ROWS: tuple[ExportRows, ...] = get_args(ExportRows.__value__)

MEDIA_TYPES: dict[ExportFormat, str] = {
    "csv": "text/csv; charset=utf-8",
    "parquet": "application/vnd.apache.parquet",
}
# Lets Excel read the CSV as UTF-8.
UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class ExportRequest:
    """What an export asks for.

    Attributes
    ----------
    file_format : ExportFormat
        ``"csv"`` or ``"parquet"``.
    rows : ExportRows
        ``"filtered"`` for every row matching the filters, on all pages, or
        ``"selected"`` for the selected rows.
    state : TableState
        Sort order and filters of the table when it was exported.
    selected : tuple[str, ...]
        Selected keys as text (``key_text``); empty unless ``rows`` is
        ``"selected"``.
    """

    file_format: ExportFormat
    rows: ExportRows
    state: TableState
    selected: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportFile:
    """An exported file.

    Attributes
    ----------
    content : bytes
        File contents.
    filename : str
        File name, ``<export_name>_<YYYYmmdd-HHMMSS>.<csv|parquet>``.
    media_type : str
        Media type of the contents.
    """

    content: bytes
    filename: str
    media_type: str

    @property
    def headers(self) -> dict[str, str]:
        """Return the response headers that download the file.

        Returns
        -------
        dict[str, str]
            ``Content-Disposition`` as an attachment with the file name, also
            percent-encoded (``filename*``) for names beyond ASCII.
        """
        fallback = self.filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
        return {
            "Content-Disposition": (
                f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(self.filename)}"
            )
        }


def _choice[T: str](raw: Sequence[str], allowed: tuple[T, ...], key: str) -> T:
    """Return the last value of a parameter that must be one of ``allowed``.

    Parameters
    ----------
    raw : Sequence[str]
        Every value of the parameter.
    allowed : tuple[T, ...]
        Values it may have.
    key : str
        Parameter name used in errors.

    Returns
    -------
    T
        The value.

    Raises
    ------
    ValueError
        If the parameter is missing or has another value.
    """
    text = raw[-1].strip() if raw else ""
    for value in allowed:
        if text == value:
            return value
    raise ValueError(f"{key!r} must be one of {', '.join(allowed)}: {text!r}")


def parse_selection(text: str) -> tuple[str, ...]:
    """Parse the selection sent as a JSON array of keys.

    Parameters
    ----------
    text : str
        Value of the hidden selection input.

    Returns
    -------
    tuple[str, ...]
        The keys.

    Raises
    ------
    ValueError
        If ``text`` is not a JSON array of strings.
    """
    try:
        keys = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"the selection must be a JSON array: {error}") from error
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError("the selection must be a JSON array of strings")
    return tuple(keys)


def parse_export(parameters: Mapping[str, Sequence[str]], config: TableConfig) -> ExportRequest:
    """Build an ``ExportRequest`` from the fields of an export.

    The fields are the table's query parameters (``parse_state``; ``page``
    is ignored), ``<table_id>.export_format`` (``csv`` or ``parquet``),
    ``<table_id>.export_rows`` (``filtered`` or ``selected``), and, for the
    selected rows, the selection as one JSON array named ``selection_name``.

    Parameters
    ----------
    parameters : Mapping[str, Sequence[str]]
        Form fields with every value of each name.
    config : TableConfig
        Table settings.

    Returns
    -------
    ExportRequest
        Parsed request.

    Raises
    ------
    ValueError
        If the format or the rows are missing or unknown, the selected rows
        of a table without selection are asked for, the selection is not a
        JSON array of strings, or ``parse_state`` rejects the other fields.
    """
    format_key = config.prefix + EXPORT_FORMAT_PARAMETER
    rows_key = config.prefix + EXPORT_ROWS_PARAMETER
    file_format = _choice(parameters.get(format_key, ()), EXPORT_FORMATS, format_key)
    rows = _choice(parameters.get(rows_key, ()), EXPORT_ROWS, rows_key)
    state = parse_state(
        {key: raw for key, raw in parameters.items() if key not in (format_key, rows_key)},
        config,
    )
    if rows == "filtered":
        return ExportRequest(file_format, rows, state)
    if not config.selectable:
        raise ValueError("a table without selection has no selected rows")
    raw = parameters.get(config.selection_name, ())
    return ExportRequest(file_format, rows, state, parse_selection(raw[-1] if raw else "[]"))


def export_rows(
    frame: pl.DataFrame | pl.LazyFrame, request: ExportRequest, config: TableConfig
) -> pl.DataFrame:
    """Return the rows and columns that an export writes.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table in its default order, as given to
        ``apply_state``.
    request : ExportRequest
        What the export asks for.
    config : TableConfig
        Table settings.

    Returns
    -------
    pl.DataFrame
        The shown columns (``TableConfig.columns``, in display order and
        under their frame names) of the rows matching the filters, on all
        pages, through the same ``filter_expression`` as the table view, or
        of every selected row, also those that the filters hide. Rows are in
        the table's sort order (``sort_frame``); selected keys missing from
        the frame are skipped.
    """
    lazy = frame.lazy()
    if request.rows == "filtered":
        rows = matching_rows(lazy, request.state, config)
    else:
        unfiltered = TableState(sort_by=request.state.sort_by, descending=request.state.descending)
        rows = matching_rows(lazy, unfiltered, config).filter(
            key_text(config).is_in(list(request.selected))
        )
    return rows.select(column.name for column in config.columns).collect()


def export_filename(config: TableConfig, file_format: ExportFormat, moment: datetime) -> str:
    """Return the name of an exported file.

    Parameters
    ----------
    config : TableConfig
        Table settings naming the file (``export_name``).
    file_format : ExportFormat
        ``"csv"`` or ``"parquet"``, the file extension.
    moment : datetime
        Time of the export.

    Returns
    -------
    str
        ``<export_name>_<YYYYmmdd-HHMMSS>.<csv|parquet>``.
    """
    return f"{config.export_name}_{moment:%Y%m%d-%H%M%S}.{file_format}"


def file_content(rows: pl.DataFrame, file_format: ExportFormat) -> bytes:
    """Write rows as the contents of a CSV or Parquet file.

    Parameters
    ----------
    rows : pl.DataFrame
        Rows to write.
    file_format : ExportFormat
        ``"csv"``: UTF-8 with a byte order mark (so that Excel reads it as
        UTF-8), a header line, dates and datetimes in ISO 8601, and null as
        an empty field. ``"parquet"``: keeps the column types.

    Returns
    -------
    bytes
        File contents.
    """
    if file_format == "csv":
        return UTF8_BOM + rows.write_csv().encode("utf-8")
    buffer = io.BytesIO()
    rows.write_parquet(buffer)
    return buffer.getvalue()


def export_file(
    frame: pl.DataFrame | pl.LazyFrame,
    request: ExportRequest,
    config: TableConfig,
    moment: datetime,
) -> ExportFile:
    """Export the rows of a table as a file.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Every row of the table in its default order.
    request : ExportRequest
        What the export asks for, for example from ``parse_export``.
    config : TableConfig
        Table settings.
    moment : datetime
        Time of the export, which names the file.

    Returns
    -------
    ExportFile
        The file of ``export_rows`` written by ``file_content``.
    """
    rows = export_rows(frame, request, config)
    return ExportFile(
        content=file_content(rows, request.file_format),
        filename=export_filename(config, request.file_format, moment),
        media_type=MEDIA_TYPES[request.file_format],
    )
