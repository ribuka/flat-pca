"""Registering catalog job results in the workspace database."""

from __future__ import annotations

from pathlib import Path

from ..database import Database, quote_identifier
from ..jobs.catalog import FILES_RESULT, METADATA_RESULT, SEGMENTS_RESULT


def _parquet_source(path: Path) -> str:
    """Return a DuckDB ``read_parquet`` call for a result file.

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    str
        SQL table expression reading ``path``.
    """
    return "read_parquet('" + str(path).replace("'", "''") + "')"


def known_files(database: Database) -> dict[str, dict[str, object]]:
    """Return the identity of every cataloged file.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    dict[str, dict[str, object]]
        ``path``, ``size``, and ``mtime_ns`` keyed by stem.
    """
    rows = database.fetch_dicts("SELECT stem, path, size, mtime_ns FROM files")
    return {
        str(row["stem"]): {
            "path": row["path"],
            "size": row["size"],
            "mtime_ns": row["mtime_ns"],
        }
        for row in rows
    }


def apply_catalog_result(
    database: Database, config: dict[str, object], run_dir: Path
) -> dict[str, object]:
    """Replace the catalog with the results of a finished catalog job.

    Files missing from the result are removed, rescanned files replace their
    previous rows, and unchanged files keep theirs. ``file_metadata`` is
    replaced by the CSV contents, or emptied when no CSV is configured.

    Parameters
    ----------
    database : Database
        Workspace database.
    config : dict[str, object]
        Job configuration built by ``build_catalog_config``.
    run_dir : Path
        Run directory written by ``run_catalog``.

    Returns
    -------
    dict[str, object]
        ``runs`` columns to store: the number of cataloged files.

    Raises
    ------
    FileNotFoundError
        If an expected result file is missing.
    """
    files_path = run_dir / FILES_RESULT
    segments_path = run_dir / SEGMENTS_RESULT
    metadata_path = run_dir / METADATA_RESULT
    expected = [files_path, segments_path]
    if config["metadata"] is not None:
        expected.append(metadata_path)
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(f"catalog result missing: {path.name}")

    files = _parquet_source(files_path)
    stale = f"stem NOT IN (SELECT stem FROM {files} WHERE NOT rescanned)"
    metadata_columns = ", ".join(
        ["stem", *(quote_identifier(name) for name in database.metadata_columns)]
    )
    with database.transaction() as connection:
        connection.execute(f"DELETE FROM files WHERE {stale}")
        connection.execute(f"DELETE FROM segments WHERE {stale}")
        connection.execute(
            "INSERT INTO files SELECT stem, path, size, mtime_ns, n_wavelengths, "
            "wavelength_min, wavelength_max, time_min, time_max, n_rows "
            f"FROM {files} WHERE rescanned"
        )
        connection.execute(
            "INSERT INTO segments SELECT stem, step, sequence, n_rows, step_time_max "
            f"FROM {_parquet_source(segments_path)}"
        )
        connection.execute("DELETE FROM file_metadata")
        if config["metadata"] is not None:
            connection.execute(
                f"INSERT INTO file_metadata SELECT {metadata_columns} "
                f"FROM {_parquet_source(metadata_path)}"
            )
        (n_files,) = connection.execute("SELECT count(*) FROM files").fetchone()
    return {"n_files": n_files}
