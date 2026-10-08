"""Choice of the data shown on the score screen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from flat_pca.feature_engineering.pca import partial_scores, time_point_order
from flat_pca.utils import natural_keys

from .component_choice import choose_component
from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError, artifact_error
from .run_dirs import RunDirs, run_dirs
from .scored_samples import choose_metadata_column, metadata_columns, scored_samples


@dataclass(frozen=True)
class ScoresRequest:
    """Choices of the score screen: the sidebar's run and files and the query.

    Attributes
    ----------
    run : str | None
        Fit or transform run shown; ``None`` for the latest succeeded fit run.
    x : int | None
        1-based component number m of the horizontal axis.
    y : int | None
        1-based component number n of the vertical axis.
    color : str | None
        Metadata column coloring the score points; ``None`` for the
        default column and ``""`` for no coloring.
    files : tuple[str, ...]
        Stems whose partial score trajectories are drawn.
    """

    run: str | None = None
    x: int | None = None
    y: int | None = None
    color: str | None = None
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScorePoints:
    """Scores of two components, one point per scored file.

    Attributes
    ----------
    samples : pl.DataFrame
        ``stem`` and the metadata columns of each scored file, in
        ``samples.parquet`` order. Files dropped by the imputation have no
        score and are absent.
    x : np.ndarray
        ``float64`` scores of component m.
    y : np.ndarray
        ``float64`` scores of component n.
    """

    samples: pl.DataFrame
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    """Partial score trajectory of one file in the PCm-PCn plane.

    Attributes
    ----------
    x : np.ndarray
        Running sums of component m after each time point.
    y : np.ndarray
        Running sums of component n after each time point.
    points : list[str]
        ``(Step, Sequence, StepTime)`` of each time point.
    """

    x: np.ndarray
    y: np.ndarray
    points: list[str]


@dataclass(frozen=True)
class ScoresView:
    """Resolved choices and the data of the score screen.

    Attributes
    ----------
    runs : list[dict[str, object]]
        Succeeded runs to choose from (see ``order_view_runs``).
    run_id : str | None
        Fit or transform run in use, or ``None`` when no fit run succeeded.
    component_count : int
        Number of components of the run.
    x : int
        Chosen component number m.
    y : int
        Chosen component number n.
    color_options : list[str]
        Metadata columns of the run's samples.
    color : str | None
        Chosen coloring column, or ``None`` for no coloring.
    file_options : list[str]
        Stems of the run's samples (its transform targets) in natural order.
    files : list[str]
        Stems whose trajectories are drawn, in option order.
    scores : ScorePoints | None
        Scores of components m and n with each file's metadata.
    trajectories : dict[str, Trajectory]
        Trajectories keyed by stem.
    dropped : list[str]
        Chosen stems whose rows the run's ``impute_strategy="drop"`` drops.
    error : str | None
        Message shown instead of the screen.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    component_count: int = 0
    x: int = 1
    y: int = 1
    color_options: list[str] = field(default_factory=list)
    color: str | None = None
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    scores: ScorePoints | None = None
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def x_name(self) -> str:
        """Return the label of component m, such as ``"PC1"``."""
        return f"PC{self.x}"

    @property
    def y_name(self) -> str:
        """Return the label of component n, such as ``"PC2"``."""
        return f"PC{self.y}"


def score_table(
    artifacts: DisplayArtifacts, columns: Sequence[str], x: int, y: int
) -> ScorePoints:
    """Return the scores of two components with each file's metadata.

    The scores are kept apart from the metadata, so a metadata column may
    have any name, including a score column's.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    columns : Sequence[str]
        Score-column names of every component (``PcaModel.pca_column_names``).
    x, y : int
        1-based component numbers m and n.

    Returns
    -------
    ScorePoints
        Scores of the scored files in ``samples.parquet`` order.

    Raises
    ------
    RunArtifactError
        If ``scores.parquet`` lacks a score column.
    """
    wanted = [columns[x - 1], columns[y - 1]]
    missing = [column for column in wanted if column not in artifacts.scores.columns]
    if missing:
        raise artifact_error(ValueError(f"scores.parquet lacks columns {missing}"))
    samples, scores = scored_samples(artifacts)
    return ScorePoints(
        samples=samples,
        x=scores[wanted[0]].cast(pl.Float64).to_numpy(),
        y=scores[wanted[1]].cast(pl.Float64).to_numpy(),
    )


def score_trajectories(
    dirs: RunDirs,
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    files: list[str],
    x: int,
    y: int,
) -> tuple[dict[str, Trajectory], list[str]]:
    """Return the partial score trajectories of the chosen files.

    Parameters
    ----------
    dirs : RunDirs
        Directories of the run's model and data.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    files : list[str]
        Chosen stems.
    x, y : int
        1-based component numbers m and n.

    Returns
    -------
    tuple[dict[str, Trajectory], list[str]]
        Trajectories keyed by stem, and the stems whose rows the imputation
        drops.

    Raises
    ------
    RunArtifactError
        If the model cannot be read.
    """
    model = cache.pca_model(dirs.model)
    order = time_point_order(model.columns)
    points = [
        f"({step}, {sequence}, {step_time:g})"
        for step, sequence, step_time in zip(
            order.steps.tolist(),
            order.sequences.tolist(),
            order.step_times.tolist(),
            strict=True,
        )
    ]
    stems = artifacts.samples["stem"].to_list()
    trajectories: dict[str, Trajectory] = {}
    dropped: list[str] = []
    # One file at a time, so only its time-point sums outlive the iteration.
    for stem in files:
        prepared = cache.prepared_row(dirs, stems.index(stem))
        if prepared.kept.size == 0:
            dropped.append(stem)
            continue
        result = partial_scores(prepared.values, model, (x, y), order)
        trajectories[stem] = Trajectory(
            x=result.scores[0, :, 0], y=result.scores[0, :, 1], points=points
        )
    return trajectories, dropped


def _run_view(
    run: dict[str, object],
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    request: ScoresRequest,
    runs: list[dict[str, object]],
    default_color: str | None,
) -> ScoresView:
    """Resolve the score screen of one fit run.

    Parameters
    ----------
    run : dict[str, object]
        The fit or transform run's ``runs`` row.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    request : ScoresRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded runs to choose from (see ``order_view_runs``).
    default_color : str | None
        Default coloring column.

    Returns
    -------
    ScoresView
        Scores and trajectories of the run.
    """
    dirs = run_dirs(run)
    component_count = artifacts.components.shape[0]
    x = choose_component(request.x, 1, component_count)
    y = choose_component(request.y, 2, component_count)
    color_options = metadata_columns(artifacts.samples)
    options = sorted(artifacts.samples["stem"].to_list(), key=natural_keys)
    wanted = set(request.files)
    files = [stem for stem in options if stem in wanted] or options[:1]
    common = {
        "runs": runs,
        "run_id": str(run["run_id"]),
        "component_count": component_count,
        "x": x,
        "y": y,
        "color_options": color_options,
        "color": choose_metadata_column(request.color, default_color, color_options),
        "file_options": options,
        "files": files,
    }
    try:
        model = cache.pca_model(dirs.model)
        scores = score_table(artifacts, model.pca_column_names, x, y)
        trajectories, dropped = score_trajectories(dirs, artifacts, cache, files, x, y)
    except RunArtifactError as error:
        return ScoresView(**common, error=str(error))  # type: ignore[arg-type]
    return ScoresView(
        **common,  # type: ignore[arg-type]
        scores=scores,
        trajectories=trajectories,
        dropped=dropped,
    )


def resolve_scores(
    cache: DisplayCache,
    fit_runs: list[dict[str, object]],
    request: ScoresRequest,
    default_color: str | None,
) -> ScoresView:
    """Resolve the requested choices and load the data they show.

    Unavailable choices fall back to their defaults, so the screen is shown
    whenever a fit run succeeded.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    fit_runs : list[dict[str, object]]
        Runs ordered by ``order_view_runs``; only succeeded ones are used.
    request : ScoresRequest
        Requested choices.
    default_color : str | None
        Default coloring column (``ui.default_color_by``).

    Returns
    -------
    ScoresView
        Resolved choices and data, or an error message.

    Raises
    ------
    ValueError
        If the run is invalid.
    """
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    by_id = {str(run["run_id"]): run for run in runs}
    if request.run is not None and request.run not in by_id:
        raise ValueError(f"succeeded fit run not found: {request.run}")
    run = by_id[request.run] if request.run is not None else (runs[0] if runs else None)
    if run is None:
        return ScoresView(runs=runs, error="No succeeded fit run.")
    try:
        artifacts = cache.display_artifacts(run_dirs(run))
    except RunArtifactError as error:
        return ScoresView(runs=runs, run_id=str(run["run_id"]), error=str(error))
    return _run_view(run, artifacts, cache, request, runs, default_color)
