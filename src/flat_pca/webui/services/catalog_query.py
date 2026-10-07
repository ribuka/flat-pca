"""Filtering, sorting, and summarizing the cataloged files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from flat_pca.utils import natural_keys

from ..database import Database, quote_identifier
from ..settings import MetadataColumnSettings

FILE_SORT_COLUMNS = ("stem", "n_steps", "n_segments", "n_rows")
EQUALS_PREFIX = "eq__"
MIN_PREFIX = "min__"
MAX_PREFIX = "max__"


@dataclass(frozen=True)
class FileQuery:
    """Filter and sort order of the file list.

    Attributes
    ----------
    text : str
        Case-insensitive substring that stems must contain.
    equals : dict[str, str]
        Required values of ``category`` metadata columns.
    ranges : dict[str, tuple[float | datetime | None, float | datetime | None]]
        Inclusive ``(lower, upper)`` bounds of ``number`` and ``datetime``
        metadata columns; ``None`` leaves a side open.
    sort_by : str
        Column to sort by: a file statistic or a metadata column.
    descending : bool
        Whether to sort in descending order.
    """

    text: str = ""
    equals: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[float | datetime | None, float | datetime | None]] = field(
        default_factory=dict
    )
    sort_by: str = "stem"
    descending: bool = False


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


def _parse_bound(
    raw: str, column: MetadataColumnSettings, name: str
) -> float | datetime | None:
    """Parse one range bound from a query parameter.

    Parameters
    ----------
    raw : str
        Parameter value; blank means an open bound.
    column : MetadataColumnSettings
        Column settings deciding the value type.
    name : str
        Column name used in errors.

    Returns
    -------
    float | datetime | None
        Parsed bound, or ``None`` for a blank value.

    Raises
    ------
    ValueError
        If the value cannot be parsed.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return (
            datetime.fromisoformat(text) if column.type == "datetime" else float(text)
        )
    except ValueError as error:
        raise ValueError(f"invalid bound for {name!r}: {raw!r}") from error


def parse_file_query(
    parameters: Mapping[str, str],
    columns: Mapping[str, MetadataColumnSettings],
) -> FileQuery:
    """Build a ``FileQuery`` from request query parameters.

    Recognized parameters are ``q`` (stem substring), ``sort``, ``order``
    (``asc`` or ``desc``), ``eq__<column>`` for category columns, and
    ``min__<column>`` / ``max__<column>`` for number and datetime columns.
    Blank values are ignored.

    Parameters
    ----------
    parameters : Mapping[str, str]
        Query parameters.
    columns : Mapping[str, MetadataColumnSettings]
        Configured metadata columns.

    Returns
    -------
    FileQuery
        Parsed query.

    Raises
    ------
    ValueError
        If a parameter names an unknown column, uses a filter that does not
        fit the column type, or has an unparsable value.
    """
    equals: dict[str, str] = {}
    lower: dict[str, float | datetime | None] = {}
    upper: dict[str, float | datetime | None] = {}
    for key, value in parameters.items():
        for prefix in (EQUALS_PREFIX, MIN_PREFIX, MAX_PREFIX):
            if not key.startswith(prefix):
                continue
            name = key.removeprefix(prefix)
            if name not in columns:
                raise ValueError(f"unknown metadata column: {name!r}")
            is_category = columns[name].type == "category"
            if is_category != (prefix == EQUALS_PREFIX):
                raise ValueError(f"filter {key!r} does not fit column type")
            if prefix == EQUALS_PREFIX:
                if value.strip():
                    equals[name] = value.strip()
            else:
                bound = _parse_bound(value, columns[name], name)
                (lower if prefix == MIN_PREFIX else upper)[name] = bound

    sort_by = parameters.get("sort", "") or "stem"
    if sort_by not in FILE_SORT_COLUMNS and sort_by not in columns:
        raise ValueError(f"unknown sort column: {sort_by!r}")
    order = parameters.get("order", "") or "asc"
    if order not in ("asc", "desc"):
        raise ValueError(f"invalid sort order: {order!r}")

    ranges = {
        name: (lower.get(name), upper.get(name))
        for name in [*lower, *upper]
        if lower.get(name) is not None or upper.get(name) is not None
    }
    return FileQuery(
        text=parameters.get("q", "").strip(),
        equals=equals,
        ranges=ranges,
        sort_by=sort_by,
        descending=order == "desc",
    )


def list_files(database: Database, query: FileQuery) -> list[dict[str, object]]:
    """Return cataloged files with their metadata, filtered and sorted.

    Parameters
    ----------
    database : Database
        Workspace database.
    query : FileQuery
        Filter and sort order. Its column names must have been validated,
        for example by ``parse_file_query``.

    Returns
    -------
    list[dict[str, object]]
        One row per file with ``stem``, ``path``, ``n_rows``, ``n_steps``,
        ``n_segments``, and every metadata column (null without a CSV row).
        Stems are compared in natural order; null sort values come last.
    """
    metadata_columns = ", ".join(
        f"m.{quote_identifier(name)}" for name in database.metadata_columns
    )
    conditions: list[str] = []
    parameters: list[object] = []
    if query.text:
        conditions.append("contains(lower(f.stem), lower(?))")
        parameters.append(query.text)
    for name, value in query.equals.items():
        conditions.append(f"m.{quote_identifier(name)} = ?")
        parameters.append(value)
    for name, (lower, upper) in query.ranges.items():
        if lower is not None:
            conditions.append(f"m.{quote_identifier(name)} >= ?")
            parameters.append(lower)
        if upper is not None:
            conditions.append(f"m.{quote_identifier(name)} <= ?")
            parameters.append(upper)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = database.fetch_dicts(
        "SELECT f.stem, f.path, f.n_rows, "
        "coalesce(s.n_steps, 0) AS n_steps, coalesce(s.n_segments, 0) AS n_segments"
        f"{', ' + metadata_columns if metadata_columns else ''} "
        "FROM files f "
        "LEFT JOIN (SELECT stem, count(DISTINCT step) AS n_steps, count(*) AS n_segments "
        "FROM segments GROUP BY stem) s USING (stem) "
        f"LEFT JOIN file_metadata m USING (stem) {where}",
        parameters,
    )
    rows.sort(key=lambda row: natural_keys(str(row["stem"])))
    present = [row for row in rows if row[query.sort_by] is not None]
    missing = [row for row in rows if row[query.sort_by] is None]
    if query.sort_by == "stem":
        if query.descending:
            present.reverse()
    else:
        present.sort(key=lambda row: row[query.sort_by], reverse=query.descending)
    return present + missing


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
