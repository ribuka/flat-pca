"""Choice of the data shown on the model screen, which depends on the run only."""

from __future__ import annotations

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

from .component_choice import choose_component
from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError
from .segment_choice import (
    choose_segment,
    feature_segments,
    format_segment,
    parse_segment,
)
from .spectral_matrix import SpectralMatrix, feature_segment_matrix

AGGREGATION_LABELS: dict[LoadingAggregation, str] = {
    "mean": "平均",
    "rms": "RMS",
    "abs_mean": "絶対値平均",
}
DEFAULT_AGGREGATION: LoadingAggregation = "rms"
COMPONENT_VALUE_NAME = "coefficient"


@dataclass(frozen=True)
class ModelRequest:
    """Choices requested by the model screen's query parameters.

    Attributes
    ----------
    run : str | None
        Fit run shown; ``None`` for the latest succeeded fit run.
    x : int | None
        1-based component number m of the loading plot's horizontal axis.
    y : int | None
        1-based component number n of the loading plot's vertical axis.
    aggregation : str | None
        A key of ``AGGREGATION_LABELS``; ``None`` for the default.
    component : int | None
        1-based component number k of the component heatmap.
    segment : str | None
        ``"{Step}:{Sequence}"`` of the component heatmap.
    """

    run: str | None = None
    x: int | None = None
    y: int | None = None
    aggregation: str | None = None
    component: int | None = None
    segment: str | None = None


@dataclass(frozen=True)
class ComponentRequest:
    """Choices that fix the matrix of the component heatmap.

    The model screen keeps the matrix it draws under this key, so its trend
    requests are cut from it.

    Attributes
    ----------
    run : str
        Fit run.
    component : int
        1-based component number k.
    segment : str
        ``"{Step}:{Sequence}"``.
    """

    run: str
    component: int
    segment: str


@dataclass(frozen=True)
class ModelView:
    """Resolved choices and the data of the model screen.

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
    aggregation : LoadingAggregation
        Chosen loading aggregation.
    component : int
        Chosen component number k.
    segment_options : list[tuple[int, int]]
        ``(Step, Sequence)`` pairs of the run's features.
    segment : tuple[int, int] | None
        Chosen pair.
    explained_variance : pl.DataFrame | None
        Result of ``PcaModel.get_explained_variance_table``.
    loadings : pl.DataFrame | None
        ``wavelength`` with the aggregated ``PC{m}`` and ``PC{n}``
        loadings.
    matrices : dict[str, SpectralMatrix]
        Unbinned matrix of component k keyed by ``PC{k}``.
    error : str | None
        Message shown instead of the screen.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    component_count: int = 0
    x: int = 1
    y: int = 1
    aggregation: LoadingAggregation = DEFAULT_AGGREGATION
    component: int = 1
    segment_options: list[tuple[int, int]] = field(default_factory=list)
    segment: tuple[int, int] | None = None
    explained_variance: pl.DataFrame | None = None
    loadings: pl.DataFrame | None = None
    matrices: dict[str, SpectralMatrix] = field(default_factory=dict)
    error: str | None = None

    @property
    def x_name(self) -> str:
        """Return the label of component m, such as ``"PC1"``."""
        return f"PC{self.x}"

    @property
    def y_name(self) -> str:
        """Return the label of component n, such as ``"PC2"``."""
        return f"PC{self.y}"


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


def component_matrices(
    artifacts: DisplayArtifacts, component: int, segment: tuple[int, int]
) -> dict[str, SpectralMatrix]:
    """Reshape one component's coefficients into a ``(Step, Sequence)`` matrix.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    component : int
        1-based component number k.
    segment : tuple[int, int]
        ``(Step, Sequence)`` to show.

    Returns
    -------
    dict[str, SpectralMatrix]
        The matrix keyed by ``PC{k}``.
    """
    # Read only this row of the memory-mapped components.
    values = np.asarray(artifacts.components[component - 1], dtype=np.float64)
    return {f"PC{component}": feature_segment_matrix(artifacts.features, values, *segment)}


def _run_view(
    run: dict[str, object],
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    request: ModelRequest,
    runs: list[dict[str, object]],
) -> ModelView:
    """Resolve the model screen of one fit run.

    Parameters
    ----------
    run : dict[str, object]
        The fit run's ``runs`` row.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model.
    request : ModelRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.

    Returns
    -------
    ModelView
        Explained variance, loadings, and component matrix of the run.
    """
    component_count = artifacts.components.shape[0]
    x = choose_component(request.x, 1, component_count)
    y = choose_component(request.y, 2, component_count)
    component = choose_component(request.component, 1, component_count)
    aggregation = cast(
        LoadingAggregation,
        request.aggregation
        if request.aggregation in LOADING_AGGREGATIONS
        else DEFAULT_AGGREGATION,
    )
    segment_options = feature_segments(artifacts)
    segment = choose_segment(segment_options, parse_segment(request.segment))
    assert segment is not None  # a fit run always has features
    common = {
        "runs": runs,
        "run_id": str(run["run_id"]),
        "component_count": component_count,
        "x": x,
        "y": y,
        "aggregation": aggregation,
        "component": component,
        "segment_options": segment_options,
        "segment": segment,
    }
    try:
        model = cache.pca_model(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return ModelView(**common, error=str(error))  # type: ignore[arg-type]
    return ModelView(
        **common,  # type: ignore[arg-type]
        explained_variance=model.get_explained_variance_table(),
        loadings=wavelength_loadings(artifacts, x, y, aggregation),
        matrices=component_matrices(artifacts, component, segment),
    )


def resolve_model(
    cache: DisplayCache,
    fit_runs: list[dict[str, object]],
    request: ModelRequest,
) -> ModelView:
    """Resolve the requested choices and load the data they show.

    Unavailable choices fall back to their defaults, so the screen is shown
    whenever a fit run succeeded.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    fit_runs : list[dict[str, object]]
        Fit runs, newest first; only succeeded ones are used.
    request : ModelRequest
        Requested choices.

    Returns
    -------
    ModelView
        Resolved choices and data, or an error message.

    Raises
    ------
    ValueError
        If the run, the loading aggregation, or the segment format is
        invalid.
    """
    if request.aggregation is not None and request.aggregation not in LOADING_AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {request.aggregation!r}")
    parse_segment(request.segment)
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    by_id = {str(run["run_id"]): run for run in runs}
    if request.run is not None and request.run not in by_id:
        raise ValueError(f"succeeded fit run not found: {request.run}")
    run = by_id[request.run] if request.run is not None else (runs[0] if runs else None)
    if run is None:
        return ModelView(runs=runs, error="成功した fit run がありません")
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return ModelView(runs=runs, run_id=str(run["run_id"]), error=str(error))
    return _run_view(run, artifacts, cache, request, runs)


def component_request(view: ModelView) -> ComponentRequest | None:
    """Return the key of the component matrix a resolved view draws.

    Parameters
    ----------
    view : ModelView
        Resolved view.

    Returns
    -------
    ComponentRequest | None
        Run, component, and segment of the drawn matrix, or ``None`` when
        the view draws none.
    """
    segment = format_segment(view.segment)
    if not view.matrices or view.run_id is None or segment is None:
        return None
    return ComponentRequest(run=view.run_id, component=view.component, segment=segment)
