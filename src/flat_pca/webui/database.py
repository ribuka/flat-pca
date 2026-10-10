"""DuckDB connection and schema of the Web UI workspace."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import duckdb
import polars as pl

from .settings import MetadataColumnSettings, MetadataColumnType

METADATA_SQL_TYPES: dict[MetadataColumnType, str] = {
    "category": "VARCHAR",
    "number": "DOUBLE",
    "datetime": "TIMESTAMP",
}

_STATIC_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS files (
        stem VARCHAR PRIMARY KEY,
        path VARCHAR NOT NULL,
        size BIGINT NOT NULL,
        mtime_ns BIGINT NOT NULL,
        n_wavelengths INTEGER NOT NULL,
        wavelength_min DOUBLE NOT NULL,
        wavelength_max DOUBLE NOT NULL,
        time_min DOUBLE NOT NULL,
        time_max DOUBLE NOT NULL,
        n_rows BIGINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS segments (
        stem VARCHAR NOT NULL,
        step BIGINT NOT NULL,
        sequence BIGINT NOT NULL,
        n_rows BIGINT NOT NULL,
        step_time_max DOUBLE NOT NULL,
        PRIMARY KEY (stem, step, sequence)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL,
        status VARCHAR NOT NULL,
        config_json VARCHAR NOT NULL,
        n_files INTEGER,
        n_features BIGINT,
        n_components INTEGER,
        duration_s DOUBLE,
        error VARCHAR,
        artifact_dir VARCHAR NOT NULL
    )
    """,
)


def quote_identifier(name: str) -> str:
    """Quote a SQL identifier for DuckDB.

    Parameters
    ----------
    name : str
        Identifier, such as a metadata column name.

    Returns
    -------
    str
        Double-quoted identifier with embedded quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


class Database:
    """Thread-safe wrapper around one DuckDB connection.

    Only the application (parent) process opens the database; job child
    processes never write to it. A single lock serializes every statement
    because request handlers and the job monitor thread share the
    connection. ``generation`` counts the statements run through
    ``execute`` and ``transaction``, so a reader can tell whether the
    tables may have changed since it last read them; ``fetch_dicts`` and
    ``fetch_frame`` are for reading the catalog and do not count.

    Parameters
    ----------
    path : Path
        DuckDB file path. Its parent directory is created if missing.
    metadata_columns : Mapping[str, MetadataColumnSettings]
        Metadata columns that ``file_metadata`` must hold besides ``stem``.
    """

    def __init__(
        self,
        path: Path,
        metadata_columns: Mapping[str, MetadataColumnSettings],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        self._lock = threading.RLock()
        self._generation = 0
        self.metadata_columns = dict(metadata_columns)
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the tables, rebuilding ``file_metadata`` if its columns changed.

        ``file_metadata`` is rebuilt (empty) when its column names, order, or
        SQL types differ from the configured metadata columns.
        """
        with self.transaction() as connection:
            for statement in _STATIC_SCHEMA:
                connection.execute(statement)
            expected = [
                ("stem", "VARCHAR"),
                *(
                    (name, METADATA_SQL_TYPES[column.type])
                    for name, column in self.metadata_columns.items()
                ),
            ]
            existing = connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'file_metadata' ORDER BY ordinal_position"
            ).fetchall()
            if existing != expected:
                connection.execute(
                    "CREATE OR REPLACE TABLE file_metadata "
                    f"({self.metadata_definition()})"
                )

    def metadata_definition(self) -> str:
        """Return the column definition list of ``file_metadata``.

        Returns
        -------
        str
            SQL column definitions, starting with ``stem``.
        """
        definitions = ["stem VARCHAR PRIMARY KEY"]
        definitions.extend(
            f"{quote_identifier(name)} {METADATA_SQL_TYPES[column.type]}"
            for name, column in self.metadata_columns.items()
        )
        return ", ".join(definitions)

    @property
    def generation(self) -> int:
        """Return the number of writing statements and transactions run so far.

        Returns
        -------
        int
            Increased by every ``execute`` and ``transaction`` (also a failed
            one), so equal values mean that no write ran in between.
        """
        return self._generation

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Run statements atomically while holding the connection lock.

        Yields
        ------
        duckdb.DuckDBPyConnection
            The locked connection inside an open transaction.
        """
        with self._lock:
            self._generation += 1
            self._connection.execute("BEGIN TRANSACTION")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")

    def execute(self, sql: str, parameters: Sequence[object] = ()) -> None:
        """Execute one statement that returns no rows.

        Parameters
        ----------
        sql : str
            SQL statement.
        parameters : Sequence[object], default ()
            Positional ``?`` parameters.
        """
        with self._lock:
            self._generation += 1
            self._connection.execute(sql, list(parameters))

    def fetch_dicts(
        self, sql: str, parameters: Sequence[object] = ()
    ) -> list[dict[str, object]]:
        """Run a query and return its rows as dictionaries.

        Parameters
        ----------
        sql : str
            SQL query.
        parameters : Sequence[object], default ()
            Positional ``?`` parameters.

        Returns
        -------
        list[dict[str, object]]
            One dictionary per row, keyed by column name.
        """
        with self._lock:
            cursor = self._connection.execute(sql, list(parameters))
            names = [description[0] for description in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def fetch_frame(self, sql: str) -> pl.DataFrame:
        """Run a query and return its rows as a polars frame.

        The rows pass from DuckDB to polars as Arrow data, without Python
        objects per value.

        Parameters
        ----------
        sql : str
            SQL query without parameters.

        Returns
        -------
        pl.DataFrame
            The rows with the column types DuckDB gives them.
        """
        with self._lock:
            return pl.DataFrame(self._connection.sql(sql))

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._connection.close()
