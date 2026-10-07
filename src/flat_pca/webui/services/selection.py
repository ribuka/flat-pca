"""The file set selected on the data selection screen."""

from __future__ import annotations

import json
import threading

from ..database import Database
from .catalog_query import existing_stems


def parse_stems_json(text: str) -> list[str]:
    """Parse a JSON array of file stems.

    The selection is sent as one form field so that its size is not bound by
    the limit on the number of form fields.

    Parameters
    ----------
    text : str
        JSON array of strings.

    Returns
    -------
    list[str]
        The stems.

    Raises
    ------
    ValueError
        If ``text`` is not a JSON array of strings.
    """
    try:
        stems = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"stems must be a JSON array: {error}") from error
    if not isinstance(stems, list) or not all(isinstance(stem, str) for stem in stems):
        raise ValueError("stems must be a JSON array of strings")
    return stems


class FileSelection:
    """Thread-safe holder of the selected file stems.

    The selection is handed from the data selection screen to the
    preprocessing screen and lives for the lifetime of the workspace.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stems: list[str] = []

    @property
    def stems(self) -> list[str]:
        """Return the selected stems.

        Returns
        -------
        list[str]
            A copy of the selected stems in natural order.
        """
        with self._lock:
            return list(self._stems)

    def replace(self, database: Database, stems: list[str]) -> list[str]:
        """Replace the selection with the cataloged stems among ``stems``.

        Parameters
        ----------
        database : Database
            Workspace database used to drop stems not in the catalog.
        stems : list[str]
            Requested stems.

        Returns
        -------
        list[str]
            The new selection.
        """
        selected = existing_stems(database, stems)
        with self._lock:
            self._stems = selected
        return list(selected)
