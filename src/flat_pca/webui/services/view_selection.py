"""The run and the files chosen in the sidebar for the display screens."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from flat_pca.utils import natural_keys

from .display_cache import DisplayCache
from .fit_artifacts import RunArtifactError
from .run_dirs import fit_run_reference, run_dirs


class ViewSelection:
    """Thread-safe holder of the run and the files chosen in the sidebar.

    The choice is shared by every display screen and every browser tab, and
    lives for the lifetime of the workspace. It is distinct from the file
    set selected for fitting (``FileSelection``). The lock is re-entrant, so
    a caller can hold ``transaction()`` while it resolves the run in use and
    updates the choice, and no other request changes the choice between.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._run_id: str | None = None
        self._stems: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the lock across several reads and updates of the choice.

        Yields
        ------
        None
            Control while the lock is held.
        """
        with self._lock:
            yield

    @property
    def run_id(self) -> str | None:
        """Return the chosen fit or transform run.

        Returns
        -------
        str | None
            The chosen run, or ``None`` for the latest succeeded fit run.
        """
        with self._lock:
            return self._run_id

    @property
    def stems(self) -> list[str]:
        """Return the chosen stems.

        Returns
        -------
        list[str]
            A copy of the chosen stems.
        """
        with self._lock:
            return list(self._stems)

    def choose_run(self, run_id: str) -> None:
        """Choose a fit or transform run, clearing the chosen files if the run changes.

        Parameters
        ----------
        run_id : str
            Chosen fit or transform run.
        """
        with self._lock:
            if run_id != self._run_id:
                self._stems = []
            self._run_id = run_id

    def replace_stems(self, stems: list[str]) -> None:
        """Replace the chosen stems.

        Parameters
        ----------
        stems : list[str]
            New stems, already validated by the caller.
        """
        with self._lock:
            self._stems = list(stems)

    def add_stem(self, stem: str, options: list[str], max_files: int) -> bool:
        """Add one stem unless the limit of chosen files is reached.

        The chosen stems that are no longer among ``options`` are dropped
        first, so they do not count toward the limit. Dropping, the limit
        check, and the addition happen under one lock, so concurrent
        additions never undo each other.

        Parameters
        ----------
        stem : str
            Stem to add, one of ``options``.
        options : list[str]
            Transform targets of the run in use.
        max_files : int
            Maximum number of chosen stems.

        Returns
        -------
        bool
            Whether ``stem`` is chosen afterwards; ``False`` if it was not
            chosen and the limit is reached.
        """
        allowed = set(options)
        with self._lock:
            self._stems = [chosen for chosen in self._stems if chosen in allowed]
            if stem in self._stems:
                return True
            if len(self._stems) >= max_files:
                return False
            self._stems.append(stem)
            return True


@dataclass(frozen=True)
class ViewChoice:
    """Resolved run and files of the sidebar.

    Attributes
    ----------
    runs : list[dict[str, object]]
        Succeeded runs to choose from: fit runs, newest first, each followed
        by its transform runs, newest first (see ``order_view_runs``).
    run_id : str | None
        Fit or transform run in use: the chosen one, or the latest succeeded
        fit run when none is chosen or the chosen one is gone. ``None``
        without runs.
    fit_run_id : str | None
        Fit run whose model the run in use transforms with: the run itself
        for a fit run. ``None`` without runs.
    file_options : list[str]
        Transform targets of the run (the stems of its ``samples.parquet``)
        in natural order; empty without a run or when its artifacts cannot
        be read.
    files : list[str]
        Chosen stems among ``file_options``, in option order.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    fit_run_id: str | None = None
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def is_transform_run(self) -> bool:
        """Return whether the run in use is a transform run."""
        return self.run_id is not None and self.run_id != self.fit_run_id


def order_view_runs(
    fit_runs: list[dict[str, object]], transform_runs: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Order the succeeded runs the sidebar chooses from.

    Parameters
    ----------
    fit_runs : list[dict[str, object]]
        Fit runs, newest first; only succeeded ones are used.
    transform_runs : list[dict[str, object]]
        Transform runs, newest first; only succeeded ones whose fit run is
        among ``fit_runs`` are used.

    Returns
    -------
    list[dict[str, object]]
        Each succeeded fit run followed by its transform runs, in the given
        orders.
    """
    by_fit_run: dict[str, list[dict[str, object]]] = {}
    for run in transform_runs:
        reference = fit_run_reference(run)
        if run["status"] == "succeeded" and reference is not None:
            by_fit_run.setdefault(reference[0], []).append(run)
    ordered: list[dict[str, object]] = []
    for run in fit_runs:
        if run["status"] == "succeeded":
            ordered.append(run)
            ordered.extend(by_fit_run.get(str(run["run_id"]), []))
    return ordered


def transform_stems(cache: DisplayCache, run: dict[str, object]) -> list[str]:
    """Return the transform targets of a fit or transform run in natural order.

    The transform targets of a fit run are the files it was fitted on.

    Parameters
    ----------
    cache : DisplayCache
        Display cache holding the run's artifacts.
    run : dict[str, object]
        The fit or transform run's ``runs`` row.

    Returns
    -------
    list[str]
        Stems of the run's ``samples.parquet``; empty if its artifacts
        cannot be read.
    """
    try:
        artifacts = cache.display_artifacts(run_dirs(run))
    except RunArtifactError:
        return []
    return sorted(artifacts.samples["stem"].to_list(), key=natural_keys)


def resolve_view_choice(
    cache: DisplayCache,
    view_runs: list[dict[str, object]],
    run_id: str | None,
    stems: list[str],
) -> ViewChoice:
    """Resolve the sidebar's chosen run and files.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    view_runs : list[dict[str, object]]
        Runs ordered by ``order_view_runs``; only succeeded ones are used.
    run_id : str | None
        Chosen run; ``None`` or a run that is gone falls back to the latest.
    stems : list[str]
        Chosen stems; those that are not transform targets of the run are
        ignored.

    Returns
    -------
    ViewChoice
        Run in use, its transform targets, and the chosen ones.
    """
    runs = [run for run in view_runs if run["status"] == "succeeded"]
    if not runs:
        return ViewChoice()
    run = next((run for run in runs if run["run_id"] == run_id), runs[0])
    options = transform_stems(cache, run)
    wanted = set(stems)
    reference = fit_run_reference(run)
    return ViewChoice(
        runs=runs,
        run_id=str(run["run_id"]),
        fit_run_id=str(run["run_id"]) if reference is None else reference[0],
        file_options=options,
        files=[stem for stem in options if stem in wanted],
    )
