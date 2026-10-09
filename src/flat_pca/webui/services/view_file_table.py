"""The shown-file dialog's table: its data table settings and its rows."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import polars as pl

from ..database import Database
from ..datatable import TableConfig
from ..settings import MetadataColumnSettings
from .file_table import file_frame, file_table_config

VIEW_FILE_TABLE_ID = "view-files"
VIEW_FILE_TABLE_URL = "/sidebar/selection/files/table"
VIEW_FILE_FORM_ID = "view-files-form"


def view_file_table_config(
    columns: Mapping[str, MetadataColumnSettings], max_files: int
) -> TableConfig:
    """Return the data table settings of the shown-file dialog's table.

    The columns, filters, and sorting are those of the data selection
    screen's file table (``file_table_config``). The table has its own id,
    since the transform screen shows both tables, and its selection input
    belongs to the dialog's form ``VIEW_FILE_FORM_ID``.

    Parameters
    ----------
    columns : Mapping[str, MetadataColumnSettings]
        Configured metadata columns.
    max_files : int
        Most files that can be shown at once (``ui.explore_max_files``).

    Returns
    -------
    TableConfig
        Settings of the table ``VIEW_FILE_TABLE_ID`` loaded from
        ``VIEW_FILE_TABLE_URL``, selecting up to ``max_files`` rows.
    """
    return dataclasses.replace(
        file_table_config(columns),
        table_id=VIEW_FILE_TABLE_ID,
        url=VIEW_FILE_TABLE_URL,
        selection_form=VIEW_FILE_FORM_ID,
        max_selected=max_files,
    )


def view_file_frame(database: Database, stems: list[str]) -> pl.DataFrame:
    """Return the cataloged rows of the given files, in their order.

    Parameters
    ----------
    database : Database
        Workspace database.
    stems : list[str]
        Transform targets of the shown run, in natural order.

    Returns
    -------
    pl.DataFrame
        One row per stem with the columns of ``file_frame``. A file missing
        from the catalog keeps its row with nulls (and a blank path).
    """
    frame = file_frame(database)
    targets = pl.DataFrame({"stem": stems}, schema={"stem": pl.String()})
    return targets.join(frame, on="stem", how="left", maintain_order="left").with_columns(
        pl.col("path").fill_null("")
    )
