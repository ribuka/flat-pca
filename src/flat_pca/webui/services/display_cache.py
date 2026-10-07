"""Least-recently-used cache of the data shown by the exploration screens."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from flat_pca.feature_engineering.flatten_pca.input import read_parquet
from flat_pca.feature_engineering.pca import PcaModel, PreparedRows, prepare_rows
from flat_pca.feature_engineering.preprocess import add_step_time_columns

from .fit_artifacts import DisplayArtifacts, load_display_artifacts, load_pca_model
from .spectral_matrix import SpectralMatrix

RAW_SPECTRA_ENTRIES = 8
FIT_ARTIFACT_ENTRIES = 4
# A restored model holds every component in float64, so keep only a few.
PCA_MODEL_ENTRIES = 2
# Unbinned matrices of shown views, one (StepTime x wavelength) float64
# matrix per file, so the trends of a shown view are cut without resolving it.
SHOWN_MATRIX_ENTRIES = 8
SHOWN_MATRIX_BYTES = 256 * 1024 * 1024


class LruCache[K: Hashable, V]:
    """Thread-safe mapping that evicts its least recently used entries.

    Parameters
    ----------
    max_entries : int
        Number of entries kept.
    max_bytes : int | None, default None
        Total size of the kept entries, measured by ``size_of``; ``None``
        for no size limit. A value larger than this is not kept at all.
    size_of : Callable[[V], int] | None, default None
        Size of one value in bytes; required with ``max_bytes``.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int | None = None,
        size_of: Callable[[V], int] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive: {max_entries}")
        if (max_bytes is None) != (size_of is None):
            raise ValueError("max_bytes and size_of must be given together")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes must be positive: {max_bytes}")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._size_of = size_of
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """Return the total size of the cached entries; 0 without ``size_of``."""
        with self._lock:
            return self._total_bytes

    def get(self, key: K) -> V | None:
        """Return the cached value of ``key`` without loading it.

        Parameters
        ----------
        key : K
            Cache key.

        Returns
        -------
        V | None
            The cached value, refreshed as the most recently used, or
            ``None`` on a miss.
        """
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key][0]

    def put(self, key: K, value: V) -> None:
        """Keep ``value`` as the most recently used entry of ``key``.

        Least recently used entries are evicted beyond ``max_entries`` or
        ``max_bytes``. A value larger than ``max_bytes`` replaces nothing
        and only drops the previous entry of ``key``.

        Parameters
        ----------
        key : K
            Cache key.
        value : V
            Value to keep.
        """
        size = 0 if self._size_of is None else self._size_of(value)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            if self.max_bytes is not None and size > self.max_bytes:
                return
            self._entries[key] = (value, size)
            self._total_bytes += size
            while len(self._entries) > self.max_entries or (
                self.max_bytes is not None and self._total_bytes > self.max_bytes
            ):
                _, (_, evicted) = self._entries.popitem(last=False)
                self._total_bytes -= evicted

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
                return self._entries[key][0]
        value = load()
        self.put(key, value)
        return value


@dataclass(frozen=True)
class ShownMatrices:
    """Unbinned matrices of one shown exploration view.

    Attributes
    ----------
    value_name : str
        Name of the shown values, used as the trends' axis title.
    matrices : dict[str, SpectralMatrix]
        Matrices keyed by trace label; the first is the heatmap.
    """

    value_name: str
    matrices: dict[str, SpectralMatrix]

    @property
    def nbytes(self) -> int:
        """Return the size of the matrices and their axes in bytes."""
        return sum(
            matrix.values.nbytes + matrix.wavelengths.nbytes + matrix.step_times.nbytes
            for matrix in self.matrices.values()
        )


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
    restore a model. The matrices of shown views are kept by their choices,
    up to ``SHOWN_MATRIX_ENTRIES`` views and ``SHOWN_MATRIX_BYTES`` in all.

    Parameters
    ----------
    prepared_row_entries : int
        Number of prepared rows kept. At least the number of files a view
        shows, so redrawing a view prepares no row again.
    """

    def __init__(self, prepared_row_entries: int) -> None:
        self._raw: LruCache[tuple[str, int, int], pl.DataFrame] = LruCache(
            RAW_SPECTRA_ENTRIES
        )
        self._runs: LruCache[str, DisplayArtifacts] = LruCache(FIT_ARTIFACT_ENTRIES)
        self._models: LruCache[str, PcaModel] = LruCache(PCA_MODEL_ENTRIES)
        self._prepared: LruCache[tuple[str, int], PreparedRows] = LruCache(
            prepared_row_entries
        )
        self._shown: LruCache[Hashable, ShownMatrices] = LruCache(
            SHOWN_MATRIX_ENTRIES, SHOWN_MATRIX_BYTES, lambda shown: shown.nbytes
        )

    @property
    def prepared_row_entries(self) -> int:
        """Return the number of prepared rows kept."""
        return self._prepared.max_entries

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

    def keep_shown_matrices(self, key: Hashable, shown: ShownMatrices) -> None:
        """Keep the matrices of a shown view, replacing those of the same key.

        Parameters
        ----------
        key : Hashable
            Choices of the view.
        shown : ShownMatrices
            The view's matrices. They are not kept when larger than
            ``SHOWN_MATRIX_BYTES``.
        """
        self._shown.put(key, shown)

    def shown_matrices(self, key: Hashable) -> ShownMatrices | None:
        """Return the kept matrices of a shown view.

        Parameters
        ----------
        key : Hashable
            Choices of the view.

        Returns
        -------
        ShownMatrices | None
            The kept matrices, or ``None`` if the view was not shown or has
            been evicted.
        """
        return self._shown.get(key)
