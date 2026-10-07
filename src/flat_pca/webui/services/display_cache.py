"""Least-recently-used cache of the data shown by the exploration screens."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path

import polars as pl

from flat_pca.feature_engineering.flatten_pca.input import read_parquet
from flat_pca.feature_engineering.pca import PcaModel, PreparedRows, prepare_rows
from flat_pca.feature_engineering.preprocess import add_step_time_columns

from .fit_artifacts import DisplayArtifacts, load_display_artifacts, load_pca_model

RAW_SPECTRA_ENTRIES = 8
FIT_ARTIFACT_ENTRIES = 4
# A restored model holds every component in float64, so keep only a few.
PCA_MODEL_ENTRIES = 2
PREPARED_ROW_ENTRIES = 16


class LruCache[K: Hashable, V]:
    """Thread-safe mapping that evicts its least recently used entries.

    Parameters
    ----------
    max_entries : int
        Number of entries kept.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive: {max_entries}")
        self.max_entries = max_entries
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)

    def get_or_load(self, key: K, load: Callable[[], V]) -> V:
        """Return the cached value of ``key``, loading it on a miss.

        The loader runs outside the lock, so concurrent misses of the same
        key may load it twice; the later value is kept.

        Parameters
        ----------
        key : K
            Cache key.
        load : Callable[[], V]
            Loads the value on a miss. Exceptions propagate and nothing is
            cached.

        Returns
        -------
        V
            The cached or newly loaded value.
        """
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        value = load()
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value


def read_raw_spectra(path: Path) -> pl.DataFrame:
    """Read one Parquet file with its ``StepTime`` columns.

    Parameters
    ----------
    path : Path
        Cataloged Parquet file.

    Returns
    -------
    pl.DataFrame
        The file's rows sorted by ``Time``, with ``StepTime`` and
        ``ReverseStepTime`` from ``add_step_time_columns``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read as Parquet.
    """
    return add_step_time_columns(read_parquet(path)).collect()  # type: ignore[union-attr]


class DisplayCache:
    """Raw spectra and fit-run artifacts kept for the exploration screens.

    Raw files are keyed by path, size, and modification time, so a changed
    file is read again. Run artifacts are keyed by run directory; their
    matrices (``X.npy`` and ``components.npy``) stay memory-mapped, so only
    the rows a view needs are read. The PCA models and the prepared rows of
    the reconstruction views are kept separately, so the other views never
    restore a model.
    """

    def __init__(self) -> None:
        self._raw: LruCache[tuple[str, int, int], pl.DataFrame] = LruCache(
            RAW_SPECTRA_ENTRIES
        )
        self._runs: LruCache[str, DisplayArtifacts] = LruCache(FIT_ARTIFACT_ENTRIES)
        self._models: LruCache[str, PcaModel] = LruCache(PCA_MODEL_ENTRIES)
        self._prepared: LruCache[tuple[str, int], PreparedRows] = LruCache(
            PREPARED_ROW_ENTRIES
        )

    def raw_spectra(self, path: Path) -> pl.DataFrame:
        """Return one file's spectra with ``StepTime`` columns.

        Parameters
        ----------
        path : Path
            Cataloged Parquet file.

        Returns
        -------
        pl.DataFrame
            See ``read_raw_spectra``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be read as Parquet.
        """
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        return self._raw.get_or_load(key, lambda: read_raw_spectra(path))

    def fit_artifacts(self, run_dir: Path) -> DisplayArtifacts:
        """Return the artifacts of one fit run.

        Parameters
        ----------
        run_dir : Path
            Run directory written by ``run_fit``.

        Returns
        -------
        DisplayArtifacts
            See ``load_display_artifacts``.

        Raises
        ------
        RunArtifactError
            If the artifacts cannot be read.
        """
        return self._runs.get_or_load(str(run_dir), lambda: load_display_artifacts(run_dir))

    def pca_model(self, run_dir: Path) -> PcaModel:
        """Return the PCA pipeline of one fit run.

        Parameters
        ----------
        run_dir : Path
            Run directory written by ``run_fit``.

        Returns
        -------
        PcaModel
            See ``load_pca_model``.

        Raises
        ------
        RunArtifactError
            If the artifacts cannot be read.
        """
        return self._models.get_or_load(str(run_dir), lambda: load_pca_model(run_dir))

    def prepared_row(self, run_dir: Path, row: int) -> PreparedRows:
        """Return one ``X.npy`` row after the run's imputation and outlier handling.

        Parameters
        ----------
        run_dir : Path
            Run directory written by ``run_fit``.
        row : int
            Zero-based row of ``X.npy``.

        Returns
        -------
        PreparedRows
            See ``prepare_rows``; ``kept`` is empty when the imputation drops
            the row.

        Raises
        ------
        RunArtifactError
            If the artifacts cannot be read.
        """

        def load() -> PreparedRows:
            # Read only this row of the memory-mapped matrix.
            values = self.fit_artifacts(run_dir).x[row : row + 1]
            return prepare_rows(values, self.pca_model(run_dir))

        return self._prepared.get_or_load((str(run_dir), row), load)
