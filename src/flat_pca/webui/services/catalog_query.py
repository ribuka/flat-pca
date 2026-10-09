"""Listing and summarizing the cataloged files."""

from __future__ import annotations

from dataclasses import dataclass

from flat_pca.utils import natural_keys

from ..database import Database, quote_identifier


@dataclass(frozen=True)
class MetadataWarnings:
    """Mismatches between cataloged files and metadata CSV rows.

    Attributes
    ----------
    files_without_metadata : list[str]
        Stems of cataloged files with no CSV row.
    metadata_without_files : list[str]
        CSV keys with no cataloged file.
    """

    files_without_metadata: list[str]
    metadata_without_files: list[str]


def list_files(database: Database) -> list[dict[str, object]]:
    """Return every cataloged file with its metadata.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    list[dict[str, object]]
        One row per file with ``stem``, ``path``, ``n_rows``, ``n_steps``,
        ``n_segments``, and every metadata column (null without a CSV row),
        with stems in natural order.
    """
    metadata_columns = ", ".join(
        f"m.{quote_identifier(name)}" for name in database.metadata_columns
    )
    rows = database.fetch_dicts(
        "SELECT f.stem, f.path, f.n_rows, "
        "coalesce(s.n_steps, 0) AS n_steps, coalesce(s.n_segments, 0) AS n_segments"
        f"{', ' + metadata_columns if metadata_columns else ''} "
        "FROM files f "
        "LEFT JOIN (SELECT stem, count(DISTINCT step) AS n_steps, count(*) AS n_segments "
        "FROM segments GROUP BY stem) s USING (stem) "
        "LEFT JOIN file_metadata m USING (stem)"
    )
    rows.sort(key=lambda row: natural_keys(str(row["stem"])))
    return rows


def category_options(database: Database) -> dict[str, list[str]]:
    """Return the distinct values of each category metadata column.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    dict[str, list[str]]
        Non-null values in natural order, keyed by category column name.
    """
    options: dict[str, list[str]] = {}
    for name, column in database.metadata_columns.items():
        if column.type != "category":
            continue
        quoted = quote_identifier(name)
        rows = database.fetch_dicts(
            f"SELECT DISTINCT {quoted} AS value FROM file_metadata WHERE {quoted} IS NOT NULL"
        )
        options[name] = sorted((str(row["value"]) for row in rows), key=natural_keys)
    return options


def metadata_warnings(database: Database) -> MetadataWarnings:
    """Compare cataloged files with the imported metadata rows.

    Parameters
    ----------
    database : Database
        Workspace database.

    Returns
    -------
    MetadataWarnings
        Stems missing on either side, in natural order.
    """

    def _stems(sql: str) -> list[str]:
        """Return the ``stem`` column of a query in natural order."""
        return sorted(
            (str(row["stem"]) for row in database.fetch_dicts(sql)), key=natural_keys
        )

    return MetadataWarnings(
        files_without_metadata=_stems(
            "SELECT stem FROM files WHERE stem NOT IN (SELECT stem FROM file_metadata)"
        ),
        metadata_without_files=_stems(
            "SELECT stem FROM file_metadata WHERE stem NOT IN (SELECT stem FROM files)"
        ),
    )


def existing_stems(database: Database, stems: list[str]) -> list[str]:
    """Return the given stems that are in the catalog.

    Parameters
    ----------
    database : Database
        Workspace database.
    stems : list[str]
        Candidate stems.

    Returns
    -------
    list[str]
        Unique cataloged stems among ``stems``, in natural order.
    """
    if not stems:
        return []
    rows = database.fetch_dicts(
        "SELECT stem FROM files WHERE list_contains(?, stem)", [list(set(stems))]
    )
    return sorted((str(row["stem"]) for row in rows), key=natural_keys)



def list_segments(database: Database, stem: str) -> list[tuple[int, int]]:
    """Return the ``(Step, Sequence)`` pairs of one cataloged file.

    Parameters
    ----------
    database : Database
        Workspace database.
    stem : str
        File stem.

    Returns
    -------
    list[tuple[int, int]]
        Pairs in ascending order; empty for an unknown stem.
    """
    rows = database.fetch_dicts(
        "SELECT step, sequence FROM segments WHERE stem = ? ORDER BY step, sequence",
        [stem],
    )
    return [(int(row["step"]), int(row["sequence"])) for row in rows]

@dataclass(frozen=True)
class SelectionRanges:
    """Steps and coordinate ranges of a set of cataloged files.

    Attributes
    ----------
    steps : list[int]
        Distinct ``Step`` values in ascending order.
    wavelength_min, wavelength_max : float | None
        Smallest and largest wavelength, or ``None`` without files.
    time_min, time_max : float | None
        Smallest and largest ``Time``, or ``None`` without files.
    step_time_max : float | None
        Largest ``StepTime`` of any ``(Step, Sequence)``, or ``None``.
    """

    steps: list[int]
    wavelength_min: float | None
    wavelength_max: float | None
    time_min: float | None
    time_max: float | None
    step_time_max: float | None


def selection_ranges(database: Database, stems: list[str]) -> SelectionRanges:
    """Return the steps and coordinate ranges of the given files.

    Parameters
    ----------
    database : Database
        Workspace database.
    stems : list[str]
        Stems of the files to summarize.

    Returns
    -------
    SelectionRanges
        Steps and ranges over the cataloged files among ``stems``.
    """
    (files,) = database.fetch_dicts(
        "SELECT min(wavelength_min) AS wavelength_min, "
        "max(wavelength_max) AS wavelength_max, min(time_min) AS time_min, "
        "max(time_max) AS time_max FROM files WHERE list_contains(?, stem)",
        [stems],
    )
    (segments,) = database.fetch_dicts(
        "SELECT list_sort(list_distinct(list(step))) AS steps, "
        "max(step_time_max) AS step_time_max FROM segments WHERE list_contains(?, stem)",
        [stems],
    )
    return SelectionRanges(
        steps=[int(step) for step in segments["steps"] or []],  # type: ignore[union-attr]
        wavelength_min=files["wavelength_min"],  # type: ignore[arg-type]
        wavelength_max=files["wavelength_max"],  # type: ignore[arg-type]
        time_min=files["time_min"],  # type: ignore[arg-type]
        time_max=files["time_max"],  # type: ignore[arg-type]
        step_time_max=segments["step_time_max"],  # type: ignore[arg-type]
    )
