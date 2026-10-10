"""The catalog's file table: its data table settings and its rows."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ..database import Database
from ..datatable import ColumnConfig, FilterKind, TableConfig
from ..settings import MetadataColumnSettings, MetadataColumnType
from .catalog_query import list_files

FILE_TABLE_ID = "files"
FILE_TABLE_URL = "/catalog/files"
FILE_PAGE_SIZE = 1000
METADATA_FILTERS: dict[MetadataColumnType, FilterKind] = {
    "category": "choice",
    "number": "number",
    "datetime": "datetime",
}
METADATA_DTYPES: dict[MetadataColumnType, pl.DataType] = {
    "category": pl.Categorical(),
    "number": pl.Float64(),
    "datetime": pl.Datetime("us"),
}


def file_table_config(columns: Mapping[str, MetadataColumnSettings]) -> TableConfig:
    """Return the data table settings of the file table.

    The table shows the file name (its tooltip is the path), each metadata
    column (values to check for ``category`` columns and lower and upper
    bounds for the others; each also with a null filter), and the numbers of
    Steps, ``(Step, Sequence)`` pairs, and rows without filters. Every
    column sorts, the file name in natural order. Above the table, the
    search box looks for words in any order and case in the file name (and
    any text column), the chips list the filters in use, and the "Columns"
    menu shows or hides columns. Each header shows the distribution of its
    column over every cataloged file. Rows are selected by stem into the hidden
    input ``stems``, ``FILE_PAGE_SIZE`` rows per page.

    Parameters
    ----------
    columns : Mapping[str, MetadataColumnSettings]
        Configured metadata columns.

    Returns
    -------
    TableConfig
        Settings of the table ``FILE_TABLE_ID`` loaded from
        ``FILE_TABLE_URL``.
    """
    return TableConfig(
        table_id=FILE_TABLE_ID,
        key="stem",
        columns=(
            ColumnConfig("stem", label="file", frame_order=True, title_column="path"),
            *(
                ColumnConfig(name, filter=METADATA_FILTERS[column.type])
                for name, column in columns.items()
            ),
            ColumnConfig("n_steps", label="Steps"),
            ColumnConfig("n_segments", label="(Step, Sequence)s"),
            ColumnConfig("n_rows", label="rows"),
        ),
        url=FILE_TABLE_URL,
        page_size=FILE_PAGE_SIZE,
        selectable=True,
        selection_name="stems",
        select_all_label="Select all filtered files",
        default_sort="stem",
        search=True,
        filter_chips=True,
        column_chooser=True,
        histograms=True,
    )


def file_frame(database: Database) -> pl.DataFrame:
    """Return every cataloged file with its metadata as a frame.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    pl.DataFrame
        The rows of ``list_files`` (stems in natural order) with typed
        columns, also without files.
    """
    schema = {
        "stem": pl.String(),
        "path": pl.String(),
        "n_rows": pl.Int64(),
        "n_steps": pl.Int64(),
        "n_segments": pl.Int64(),
        **{
            name: METADATA_DTYPES[column.type]
            for name, column in database.metadata_columns.items()
        },
    }
    return pl.DataFrame(list_files(database), schema=schema)
