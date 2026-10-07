"""Choice of the data shown on the score and loading screen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from flat_pca.feature_engineering.flatten_pca import (
    LOADING_AGGREGATIONS,
    LoadingAggregation,
    aggregate_loadings_by_wavelength,
)
from flat_pca.feature_engineering.pca import partial_scores
from flat_pca.spectral.schema import SOURCE_COLUMN

from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError, artifact_error

AGGREGATION_LABELS: dict[LoadingAggregation, str] = {
    "mean": "平均",
    "rms": "RMS",
    "abs_mean": "絶対値平均",
}
DEFAULT_AGGREGATION: LoadingAggregation = "rms"
# Columns of samples.parquet that are not metadata.
SAMPLE_KEY_COLUMNS = (SOURCE_COLUMN, "stem")


@dataclass(frozen=True)
class ScoresRequest:
    """Choices requested by the score screen's query parameters.

    Attributes
    ----------
    run : str | None
        Fit run shown; ``None`` for the latest succeeded fit run.
    x : int | None
        1-based component number m of the horizontal axis.
    y : int | None
        1-based component number n of the vertical axis.
    color : str | None
        Metadata column coloring the score points; ``None`` for the
        default column and ``""`` for no coloring.
    aggregation : str | None
        A key of ``AGGREGATION_LABELS``; ``None`` for the default.
    files : tuple[str, ...]
        Stems whose partial score trajectories are drawn.
    """

    run: str | None = None
    x: int | None = None
    y: int | None = None
    color: str | None = None
    aggregation: str | None = None
    files: tuple[str, ...] = ()


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
        Succeeded fit runs, newest first.
    run_id : str | None
        Fit run in use, or ``None`` when no fit run succeeded.
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
    aggregation : LoadingAggregation
        Chosen loading aggregation.
    file_options : list[str]
        Stems of the run's samples.
    files : list[str]
        Stems whose trajectories are drawn, in option order.
    scores : pl.DataFrame | None
        ``stem``, the metadata columns, and the ``PC{m}`` and ``PC{n}``
        scores of the files kept by the imputation, in sample order.
    loadings : pl.DataFrame | None
        ``wavelength`` with the aggregated ``PC{m}`` and ``PC{n}``
        loadings.
    explained_variance : pl.DataFrame | None
        Result of ``PcaModel.get_explained_variance_table``.
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
    aggregation: LoadingAggregation = DEFAULT_AGGREGATION
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    scores: pl.DataFrame | None = None
    loadings: pl.DataFrame | None = None
    explained_variance: pl.DataFrame | None = None
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


def _choose_component(requested: int | None, default: int, component_count: int) -> int:
    """Return the requested component number if available, otherwise ``default``.

    Parameters
    ----------
    requested : int | None
        Requested 1-based component number.
    default : int
        Component number used without a valid request; it is capped at
        ``component_count``.
    component_count : int
        Number of components of the run.

    Returns
    -------
    int
        Chosen 1-based component number.
    """
    if requested is not None and 1 <= requested <= component_count:
        return requested
    return min(default, component_count)


def _choose_color(
    requested: str | None, default: str | None, options: list[str]
) -> str | None:
    """Return the coloring column.

    Parameters
    ----------
    requested : str | None
        Requested column; ``""`` for no coloring and ``None`` for the
        default.
    default : str | None
        Default column (``ui.default_color_by``).
    options : list[str]
        Metadata columns of the run's samples.

    Returns
    -------
    str | None
        The requested column if available, otherwise the default column if
        available, otherwise ``None``.
    """
    if requested == "":
        return None
    for candidate in (requested, default):
        if candidate in options:
            return candidate
    return None


def score_table(
    artifacts: DisplayArtifacts, columns: Sequence[str], x: int, y: int
) -> pl.DataFrame:
    """Return the scores of two components with each file's metadata.

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
    pl.DataFrame
        ``stem``, the metadata columns, ``PC{m}``, and ``PC{n}``, one row per
        scored file in ``samples.parquet`` order. Files dropped by the
        imputation have no score and are absent.

    Raises
    ------
    RunArtifactError
        If ``scores.parquet`` lacks a score column.
    """
    wanted = {f"PC{x}": columns[x - 1], f"PC{y}": columns[y - 1]}
    missing = [column for column in wanted.values() if column not in artifacts.scores.columns]
    if missing:
        raise artifact_error(ValueError(f"scores.parquet lacks columns {missing}"))
    scores = artifacts.scores.select(
        SOURCE_COLUMN,
        *(pl.col(column).cast(pl.Float64).alias(name) for name, column in wanted.items()),
    )
    return (
        artifacts.samples.with_row_index("__order")
        .join(scores, on=SOURCE_COLUMN, how="inner")
        .sort("__order")
        .drop("__order", SOURCE_COLUMN)
    )


def wavelength_loadings(
    artifacts: DisplayArtifacts, x: int, y: int, method: LoadingAggregation
) -> pl.DataFrame:
    """Return two components' coefficients aggregated by wavelength.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    x, y : int
        1-based component numbers m and n.
    method : LoadingAggregation
        Aggregation over the features of each wavelength.

    Returns
    -------
    pl.DataFrame
        ``wavelength``, ``PC{m}``, and ``PC{n}``, one row per wavelength in
        ascending order.
    """
    long = pl.concat(
        [
            artifacts.features.select("wavelength").with_columns(
                pl.lit(component, dtype=pl.Int64).alias("component"),
                # Read only this row of the memory-mapped components.
                pl.Series(
                    "coefficient",
                    np.asarray(artifacts.components[component - 1], dtype=np.float64),
                ),
            )
            for component in sorted({x, y})
        ]
    )
    aggregated = aggregate_loadings_by_wavelength(long, method)
    by_component = {
        component: aggregated.filter(pl.col("component") == component).select(
            "wavelength", pl.col("loading").alias(f"PC{component}")
        )
        for component in {x, y}
    }
    loadings = by_component[x]
    if y != x:
        loadings = loadings.join(by_component[y], on="wavelength", how="inner")
    else:
        loadings = loadings.with_columns(pl.col(f"PC{x}").alias(f"PC{y}"))
    return loadings.sort("wavelength")


def score_trajectories(
    run_dir: Path,
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    files: list[str],
    x: int,
    y: int,
) -> tuple[dict[str, Trajectory], list[str]]:
    """Return the partial score trajectories of the chosen files.

    Parameters
    ----------
    run_dir : Path
        Run directory of the fit run.
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
    model = cache.pca_model(run_dir)
    stems = artifacts.samples["stem"].to_list()
    kept: list[str] = []
    rows: list[np.ndarray] = []
    dropped: list[str] = []
    for stem in files:
        prepared = cache.prepared_row(run_dir, stems.index(stem))
        if prepared.kept.size == 0:
            dropped.append(stem)
            continue
        kept.append(stem)
        rows.append(prepared.values)
    if not kept:
        return {}, dropped
    result = partial_scores(np.vstack(rows), model, (x, y))
    points = [
        f"({step}, {sequence}, {step_time:g})"
        for step, sequence, step_time in zip(
            result.steps.tolist(),
            result.sequences.tolist(),
            result.step_times.tolist(),
            strict=True,
        )
    ]
    trajectories = {
        stem: Trajectory(
            x=result.scores[index, :, 0], y=result.scores[index, :, 1], points=points
        )
        for index, stem in enumerate(kept)
    }
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
        The fit run's ``runs`` row.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    request : ScoresRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.
    default_color : str | None
        Default coloring column.

    Returns
    -------
    ScoresView
        Scores, loadings, explained variance, and trajectories of the run.
    """
    run_dir = Path(str(run["artifact_dir"]))
    component_count = artifacts.components.shape[0]
    x = _choose_component(request.x, 1, component_count)
    y = _choose_component(request.y, 2, component_count)
    color_options = [
        column for column in artifacts.samples.columns if column not in SAMPLE_KEY_COLUMNS
    ]
    aggregation = cast(
        LoadingAggregation,
        request.aggregation
        if request.aggregation in LOADING_AGGREGATIONS
        else DEFAULT_AGGREGATION,
    )
    stems = artifacts.samples["stem"].to_list()
    wanted = set(request.files)
    files = [stem for stem in stems if stem in wanted] or stems[:1]
    common = {
        "runs": runs,
        "run_id": str(run["run_id"]),
        "component_count": component_count,
        "x": x,
        "y": y,
        "color_options": color_options,
        "color": _choose_color(request.color, default_color, color_options),
        "aggregation": aggregation,
        "file_options": stems,
        "files": files,
    }
    try:
        model = cache.pca_model(run_dir)
        scores = score_table(artifacts, model.pca_column_names, x, y)
        trajectories, dropped = score_trajectories(run_dir, artifacts, cache, files, x, y)
    except RunArtifactError as error:
        return ScoresView(**common, error=str(error))  # type: ignore[arg-type]
    return ScoresView(
        **common,  # type: ignore[arg-type]
        scores=scores,
        loadings=wavelength_loadings(artifacts, x, y, aggregation),
        explained_variance=model.get_explained_variance_table(),
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
        Fit runs, newest first; only succeeded ones are used.
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
        If the run or the loading aggregation is invalid.
    """
    if request.aggregation is not None and request.aggregation not in LOADING_AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {request.aggregation!r}")
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    by_id = {str(run["run_id"]): run for run in runs}
    if request.run is not None and request.run not in by_id:
        raise ValueError(f"succeeded fit run not found: {request.run}")
    run = by_id[request.run] if request.run is not None else (runs[0] if runs else None)
    if run is None:
        return ScoresView(runs=runs, error="成功した fit run がありません")
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return ScoresView(runs=runs, run_id=str(run["run_id"]), error=str(error))
    return _run_view(run, artifacts, cache, request, runs, default_color)
