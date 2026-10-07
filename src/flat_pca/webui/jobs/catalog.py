"""Catalog job: scan Parquet files and the metadata CSV into run files.

The job runs in a child process and never touches the database. It writes
``files.parquet``, ``segments.parquet``, and (with a metadata CSV)
``metadata.parquet`` to the run directory; the app process registers them
with ``services.catalog_store.apply_catalog_result``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl
from loguru import logger

from flat_pca.feature_engineering.flatten_pca.input import (
    read_parquet,
    resolve_and_check_paths,
    validate_frame_schema,
)
from flat_pca.feature_engineering.preprocess.step_time import add_step_time_columns
from flat_pca.spectral.schema import METADATA_COLUMNS, parse_wavelength

from ..services.metadata_csv import read_metadata_csv
from ..settings import MetadataSettings, Settings
from .progress import write_progress

FILES_RESULT = "files.parquet"
SEGMENTS_RESULT = "segments.parquet"
METADATA_RESULT = "metadata.parquet"

FILE_SCHEMA = {
    "stem": pl.String,
    "path": pl.String,
    "size": pl.Int64,
    "mtime_ns": pl.Int64,
    "rescanned": pl.Boolean,
    "n_wavelengths": pl.Int32,
    "wavelength_min": pl.Float64,
    "wavelength_max": pl.Float64,
    "time_min": pl.Float64,
    "time_max": pl.Float64,
    "n_rows": pl.Int64,
}
SEGMENT_SCHEMA = {
    "stem": pl.String,
    "step": pl.Int64,
    "sequence": pl.Int64,
    "n_rows": pl.Int64,
    "step_time_max": pl.Float64,
}


def build_catalog_config(
    settings: Settings,
    known_files: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build the JSON configuration of a catalog job.

    Parameters
    ----------
    settings : Settings
        Application settings.
    known_files : Mapping[str, Mapping[str, object]]
        Files already in the catalog, keyed by stem, each with ``path``,
        ``size``, and ``mtime_ns``.

    Returns
    -------
    dict[str, object]
        Configuration passed to ``run_catalog``.
    """
    return {
        "root": str(settings.data.root),
        "glob": settings.data.glob,
        "metadata": None
        if settings.metadata is None
        else settings.metadata.model_dump(mode="json"),
        "known_files": {stem: dict(row) for stem, row in known_files.items()},
    }


def scan_parquet_file(path: Path) -> tuple[dict[str, object], pl.DataFrame]:
    """Summarize one Parquet file from its schema and metadata columns.

    Only ``Time``, ``Step``, and ``Sequence`` values are read; spectral
    columns are inspected through the schema alone.

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    tuple[dict[str, object], pl.DataFrame]
        File summary (wavelength count, minimum, and maximum, ``Time``
        minimum and maximum, and row count) and
        one row per ``(Step, Sequence)`` with its row count and maximum
        ``StepTime``.

    Raises
    ------
    ValueError
        If the file cannot be read or violates the input schema.
    """
    frame = read_parquet(path)
    wavelengths = [
        parse_wavelength(column) for column in validate_frame_schema(path, frame)
    ]
    metadata = frame.select(METADATA_COLUMNS).collect()
    segments = (
        add_step_time_columns(metadata)
        .group_by("Step", "Sequence")
        .agg(pl.len().alias("n_rows"), pl.col("StepTime").max().alias("step_time_max"))
        .select(
            pl.lit(path.stem).alias("stem"),
            pl.col("Step").cast(pl.Int64).alias("step"),
            pl.col("Sequence").cast(pl.Int64).alias("sequence"),
            pl.col("n_rows").cast(pl.Int64),
            pl.col("step_time_max").cast(pl.Float64),
        )
        .sort("step", "sequence")
    )
    summary = {
        "n_wavelengths": len(wavelengths),
        "wavelength_min": min(wavelengths),
        "wavelength_max": max(wavelengths),
        "time_min": float(metadata["Time"].cast(pl.Float64).min()),
        "time_max": float(metadata["Time"].cast(pl.Float64).max()),
        "n_rows": metadata.height,
    }
    return summary, segments


def discover_parquet_files(root: Path, pattern: str) -> list[Path]:
    """List Parquet files under the data root, requiring unique stems.

    Parameters
    ----------
    root : Path
        Data root directory.
    pattern : str
        Glob pattern relative to ``root``.

    Returns
    -------
    list[Path]
        Absolute file paths sorted by path text.

    Raises
    ------
    FileNotFoundError
        If ``root`` is not a directory.
    ValueError
        If two files share a ``Path.stem``.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {root}")
    paths = [path for path in root.glob(pattern) if path.is_file()]
    if not paths:
        return []
    return resolve_and_check_paths(paths, stem_uniqueness="error")


def _is_unchanged(
    path: Path, size: int, mtime_ns: int, known: Mapping[str, object] | None
) -> bool:
    """Return whether a file matches its previous catalog entry.

    Parameters
    ----------
    path : Path
        Current file path.
    size : int
        Current size in bytes.
    mtime_ns : int
        Current modification time in nanoseconds.
    known : Mapping[str, object] | None
        Previous entry for the same stem, if any.

    Returns
    -------
    bool
        ``True`` if path, size, and modification time are unchanged.
    """
    return (
        known is not None
        and known.get("path") == str(path)
        and known.get("size") == size
        and known.get("mtime_ns") == mtime_ns
    )


def run_catalog(config: dict[str, object], run_dir: Path) -> None:
    """Scan new and changed Parquet files and read the metadata CSV.

    Files whose path, size, and modification time match ``known_files`` are
    listed without being read again.

    Parameters
    ----------
    config : dict[str, object]
        Configuration built by ``build_catalog_config``.
    run_dir : Path
        Run directory receiving the result files and ``progress.json``.
    """
    known_files: Mapping[str, Mapping[str, object]] = config["known_files"]  # type: ignore[assignment]
    paths = discover_parquet_files(Path(str(config["root"])), str(config["glob"]))
    logger.info(f"found {len(paths)} Parquet files")

    file_rows: list[dict[str, object]] = []
    segment_frames = [pl.DataFrame(schema=SEGMENT_SCHEMA)]
    write_progress(run_dir, "scan", 0, len(paths))
    for index, path in enumerate(paths, start=1):
        stat = path.stat()
        row: dict[str, object] = {
            "stem": path.stem,
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if _is_unchanged(
            path, stat.st_size, stat.st_mtime_ns, known_files.get(path.stem)
        ):
            row["rescanned"] = False
        else:
            summary, segments = scan_parquet_file(path)
            row.update(summary, rescanned=True)
            segment_frames.append(segments)
        file_rows.append(row)
        write_progress(run_dir, "scan", index, len(paths))

    files = pl.DataFrame(file_rows, schema=FILE_SCHEMA)
    logger.info(f"rescanned {files['rescanned'].sum()} of {files.height} files")
    files.write_parquet(run_dir / FILES_RESULT)
    pl.concat(segment_frames).write_parquet(run_dir / SEGMENTS_RESULT)

    if config["metadata"] is not None:
        write_progress(run_dir, "metadata", 0, 1)
        metadata = MetadataSettings.model_validate(config["metadata"])
        read_metadata_csv(metadata).write_parquet(run_dir / METADATA_RESULT)
        write_progress(run_dir, "metadata", 1, 1)
