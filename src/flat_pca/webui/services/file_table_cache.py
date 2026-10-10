"""The catalog's file table read once per change of the database."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import polars as pl

from ..database import Database
from ..datatable import (
    TableConfig,
    TableState,
    TableSummary,
    TableView,
    apply_state,
    summarize_table,
)
from .catalog_query import MetadataWarnings, category_options, metadata_warnings
from .file_table import file_frame


@dataclass(frozen=True)
class CatalogSnapshot:
    """The file table's rows and what is counted over them, for one catalog.

    Attributes
    ----------
    generation : int
        ``Database.generation`` read before the rows.
    frame : pl.DataFrame
        ``file_frame`` of the database.
    options : dict[str, list[str]]
        ``category_options`` of the database.
    warnings : MetadataWarnings
        ``metadata_warnings`` of the database.
    summaries : dict[TableConfig, TableSummary]
        ``summarize_table`` of ``frame`` for each table settings asked for so
        far (``snapshot_summary``).
    """

    generation: int
    frame: pl.DataFrame
    options: dict[str, list[str]]
    warnings: MetadataWarnings
    summaries: dict[TableConfig, TableSummary] = field(default_factory=dict)


_snapshots: WeakKeyDictionary[Database, CatalogSnapshot] = WeakKeyDictionary()
_lock = threading.Lock()


def catalog_snapshot(database: Database) -> CatalogSnapshot:
    """Return the file table's rows, read again only after the database changed.

    The catalog changes only by writes (a catalog update, for example), so
    the rows, the choices, and the warnings are kept until the database's
    ``generation`` moves on; sorting, filtering, and paging the table read
    nothing again.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    CatalogSnapshot
        The kept snapshot, or a new one if a write ran since it was read.
    """
    with _lock:
        snapshot = _snapshots.get(database)
        generation = database.generation
        if snapshot is None or snapshot.generation != generation:
            snapshot = CatalogSnapshot(
                generation=generation,
                frame=file_frame(database),
                options=category_options(database),
                warnings=metadata_warnings(database),
            )
            _snapshots[database] = snapshot
        return snapshot


def snapshot_summary(snapshot: CatalogSnapshot, config: TableConfig) -> TableSummary:
    """Return the summary of a snapshot's rows for a table, made once per settings.

    Parameters
    ----------
    snapshot : CatalogSnapshot
        Snapshot from ``catalog_snapshot``.
    config : TableConfig
        Settings of a table showing every row of the snapshot.

    Returns
    -------
    TableSummary
        ``summarize_table`` of the snapshot's frame with its category
        options, kept in the snapshot.
    """
    with _lock:
        summary = snapshot.summaries.get(config)
        if summary is None:
            summary = summarize_table(snapshot.frame, config, snapshot.options)
            snapshot.summaries[config] = summary
        return summary


def file_table_view(database: Database, config: TableConfig, state: TableState) -> TableView:
    """Return one view of the file table from the kept snapshot of the catalog.

    Parameters
    ----------
    database : Database
        Workspace database.
    config : TableConfig
        Settings of the file table (``file_table_config``).
    state : TableState
        Sort order, filters, page, and page size.

    Returns
    -------
    TableView
        ``apply_state`` of the snapshot's frame with its kept summary, so
        that only the filtering, sorting, and paging run per request.
    """
    snapshot = catalog_snapshot(database)
    return apply_state(snapshot.frame, state, config, snapshot_summary(snapshot, config))


def file_table_context(
    database: Database, config: TableConfig, state: TableState, with_warnings: bool
) -> dict[str, object]:
    """Return what ``partials/file_table.html`` shows for one state of the file table.

    Parameters
    ----------
    database : Database
        Workspace database.
    config : TableConfig
        Settings of the file table.
    state : TableState
        Sort order, filters, page, and page size.
    with_warnings : bool
        Whether the metadata warnings are shown (when a metadata CSV is
        configured).

    Returns
    -------
    dict[str, object]
        ``table`` (the settings), ``view`` (``file_table_view``), and
        ``warnings`` (the kept snapshot's metadata warnings, or ``None``
        without ``with_warnings``).
    """
    return {
        "table": config,
        "view": file_table_view(database, config, state),
        "warnings": catalog_snapshot(database).warnings if with_warnings else None,
    }
