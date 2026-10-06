"""Reading the metadata CSV joined to Parquet files by ``Path.stem``."""

from __future__ import annotations

import polars as pl

from ..settings import MetadataColumnSettings, MetadataSettings

STEM_COLUMN = "stem"


def _typed_column(name: str, column: MetadataColumnSettings) -> pl.Expr:
    """Return an expression converting one text CSV column to its type.

    Parameters
    ----------
    name : str
        CSV column name.
    column : MetadataColumnSettings
        Configured column type.

    Returns
    -------
    pl.Expr
        Strict conversion expression aliased to ``name``.
    """
    text = pl.col(name).str.strip_chars()
    if column.type == "number":
        return text.cast(pl.Float64, strict=True).alias(name)
    if column.type == "datetime":
        return text.str.strptime(pl.Datetime("us"), column.format, strict=True).alias(
            name
        )
    return text.alias(name)


def read_metadata_csv(metadata: MetadataSettings) -> pl.DataFrame:
    """Read the metadata CSV and convert the configured columns.

    Parameters
    ----------
    metadata : MetadataSettings
        Metadata CSV settings.

    Returns
    -------
    pl.DataFrame
        ``stem`` (the key column) followed by the configured columns in
        settings order. Empty cells are null.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If a configured column is missing, a value cannot be converted, or a
        key is empty or repeated.
    """
    if not metadata.csv.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {metadata.csv}")
    raw = pl.read_csv(metadata.csv, infer_schema=False)
    missing = [
        name for name in [metadata.key, *metadata.columns] if name not in raw.columns
    ]
    if missing:
        raise ValueError(f"metadata CSV columns missing in {metadata.csv}: {missing}")

    try:
        frame = raw.select(
            pl.col(metadata.key).str.strip_chars().alias(STEM_COLUMN),
            *(_typed_column(name, column) for name, column in metadata.columns.items()),
        )
    except pl.exceptions.PolarsError as error:
        raise ValueError(
            f"invalid metadata CSV values in {metadata.csv}: {error}"
        ) from error

    stems = frame[STEM_COLUMN]
    if stems.is_null().any() or (stems == "").any():
        raise ValueError(f"metadata CSV key {metadata.key!r} has empty values")
    duplicated = sorted(stems.filter(stems.is_duplicated()).unique().to_list())
    if duplicated:
        raise ValueError(f"metadata CSV key {metadata.key!r} repeats: {duplicated}")
    return frame
