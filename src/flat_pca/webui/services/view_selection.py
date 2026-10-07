"""The fit run and the files chosen in the sidebar for the display screens."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from flat_pca.utils import natural_keys

from .display_cache import DisplayCache
from .fit_artifacts import RunArtifactError


class ViewSelection:
    """Thread-safe holder of the fit run and the files chosen in the sidebar.

    The choice is shared by every display screen and every browser tab, and
    lives for the lifetime of the workspace. It is distinct from the file
    set selected for fitting (``FileSelection``).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._stems: list[str] = []

    @property
    def run_id(self) -> str | None:
        """Return the chosen fit run.

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
        """Choose a fit run, clearing the chosen files if the run changes.

        Parameters
        ----------
        run_id : str
            Chosen fit run.
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
    """Resolved fit run and files of the sidebar.

    Attributes
    ----------
    runs : list[dict[str, object]]
        Succeeded fit runs to choose from, newest first.
    run_id : str | None
        Fit run in use: the chosen one, or the latest succeeded one when
        none is chosen or the chosen one is gone. ``None`` without runs.
    file_options : list[str]
        Transform targets of the run (the stems of its ``samples.parquet``)
        in natural order; empty without a run or when its artifacts cannot
        be read.
    files : list[str]
        Chosen stems among ``file_options``, in option order.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def transform_stems(cache: DisplayCache, run: dict[str, object]) -> list[str]:
    """Return the transform targets of a fit run in natural order.

    The transform targets are the files the run was fitted on.

    Parameters
    ----------
    cache : DisplayCache
        Display cache holding the run's artifacts.
    run : dict[str, object]
        The fit run's ``runs`` row.

    Returns
    -------
    list[str]
        Stems of the run's ``samples.parquet``; empty if its artifacts
        cannot be read.
    """
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError:
        return []
    return sorted(artifacts.samples["stem"].to_list(), key=natural_keys)


def resolve_view_choice(
    cache: DisplayCache,
    fit_runs: list[dict[str, object]],
    run_id: str | None,
    stems: list[str],
) -> ViewChoice:
    """Resolve the sidebar's chosen run and files.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    fit_runs : list[dict[str, object]]
        Fit runs, newest first; only succeeded ones are used.
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
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    if not runs:
        return ViewChoice()
    run = next((run for run in runs if run["run_id"] == run_id), runs[0])
    options = transform_stems(cache, run)
    wanted = set(stems)
    return ViewChoice(
        runs=runs,
        run_id=str(run["run_id"]),
        file_options=options,
        files=[stem for stem in options if stem in wanted],
    )
