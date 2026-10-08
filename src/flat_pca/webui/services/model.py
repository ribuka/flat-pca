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
from flat_pca.feature_engineering.pca import PcaModel

from .component_choice import choose_component
from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError
from .preprocess_parameters import (
    FIXED_PARAMETER_LABELS,
    PARAMETER_VALUE_NAME,
    is_parameter_key,
    parameter_options,
    parameter_values,
)
from .segment_choice import (
    choose_segment,
    feature_segments,
    format_segment,
    parse_segment,
)
from .spectral_matrix import SpectralMatrix, feature_segment_matrix

AGGREGATION_LABELS: dict[LoadingAggregation, str] = {
    "mean": "mean",
    "rms": "RMS",
    "abs_mean": "mean absolute",
}
DEFAULT_AGGREGATION: LoadingAggregation = "rms"
COMPONENT_VALUE_NAME = "coefficient"
COMPONENT_VIEW = "component"
COMPONENT_VIEW_LABEL = "PCA component k"


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
    view : str | None
        Values of the heatmap: ``COMPONENT_VIEW`` or a preprocessing
        parameter key; ``None`` for ``COMPONENT_VIEW``.
    component : int | None
        1-based component number k of the component heatmap.
    segment : str | None
        ``"{Step}:{Sequence}"`` of the heatmap.
    """

    run: str | None = None
    x: int | None = None
    y: int | None = None
    aggregation: str | None = None
    view: str | None = None
    component: int | None = None
    segment: str | None = None


@dataclass(frozen=True)
class HeatmapRequest:
    """Choices that fix the matrix of the heatmap.

    The model screen keeps the matrix it draws under this key, so its trend
    requests are cut from it.

    Attributes
    ----------
    run : str
        Fit run.
    view : str
        ``COMPONENT_VIEW`` or a preprocessing parameter key.
    component : int
        1-based component number k.
    segment : str
        ``"{Step}:{Sequence}"``.
    """

    run: str
    view: str
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
    view_options : dict[str, str]
        Labels of the heatmap's values keyed by ``COMPONENT_VIEW`` and the
        preprocessing parameter keys the run holds.
    view : str
        Chosen key of ``view_options``.
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
        Unbinned matrix of the heatmap, keyed by ``PC{k}`` or the
        parameter's label.
    notice : str | None
        Message telling that the requested parameter is absent from the run.
    error : str | None
        Message shown instead of the screen.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    component_count: int = 0
    x: int = 1
    y: int = 1
    aggregation: LoadingAggregation = DEFAULT_AGGREGATION
    view_options: dict[str, str] = field(
        default_factory=lambda: {COMPONENT_VIEW: COMPONENT_VIEW_LABEL}
    )
    view: str = COMPONENT_VIEW
    component: int = 1
    segment_options: list[tuple[int, int]] = field(default_factory=list)
    segment: tuple[int, int] | None = None
    explained_variance: pl.DataFrame | None = None
    loadings: pl.DataFrame | None = None
    matrices: dict[str, SpectralMatrix] = field(default_factory=dict)
    notice: str | None = None
    error: str | None = None

    @property
    def value_name(self) -> str:
        """Return the name of the heatmap's values, shown on its color bar."""
        return COMPONENT_VALUE_NAME if self.view == COMPONENT_VIEW else PARAMETER_VALUE_NAME

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


def parameter_matrices(
    artifacts: DisplayArtifacts,
    model: PcaModel,
    key: str,
    label: str,
    segment: tuple[int, int],
) -> dict[str, SpectralMatrix]:
    """Reshape one preprocessing parameter into a ``(Step, Sequence)`` matrix.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    model : PcaModel
        The run's PCA model, whose columns follow ``artifacts.features``.
    key : str
        A key of ``parameter_options(model)``.
    label : str
        Label of the parameter.
    segment : tuple[int, int]
        ``(Step, Sequence)`` to show.

    Returns
    -------
    dict[str, SpectralMatrix]
        The matrix keyed by ``label``.
    """
    values = parameter_values(model, key)
    return {label: feature_segment_matrix(artifacts.features, values, *segment)}


def _choose_view(requested: str | None, options: dict[str, str]) -> tuple[str, str | None]:
    """Return the requested heatmap values if the run holds them.

    Parameters
    ----------
    requested : str | None
        ``COMPONENT_VIEW``, a preprocessing parameter key, or ``None``.
    options : dict[str, str]
        Available values keyed as ``ModelView.view_options``.

    Returns
    -------
    tuple[str, str | None]
        The chosen key, ``COMPONENT_VIEW`` when the run lacks the requested
        parameter, and the message telling so.
    """
    if requested is None or requested in options:
        return requested or COMPONENT_VIEW, None
    label = FIXED_PARAMETER_LABELS.get(requested, requested)
    return (
        COMPONENT_VIEW,
        (
            f"This run has no \"{label}\" (its preprocessing settings do not keep it). "
            f"Showing {COMPONENT_VIEW_LABEL} instead."
        ),
    )


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
        Explained variance, loadings, and heatmap matrix of the run.
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
    view_options = {COMPONENT_VIEW: COMPONENT_VIEW_LABEL} | parameter_options(model)
    view, notice = _choose_view(request.view, view_options)
    matrices = (
        component_matrices(artifacts, component, segment)
        if view == COMPONENT_VIEW
        else parameter_matrices(artifacts, model, view, view_options[view], segment)
    )
    return ModelView(
        **common,  # type: ignore[arg-type]
        view_options=view_options,
        view=view,
        explained_variance=model.get_explained_variance_table(),
        loadings=wavelength_loadings(artifacts, x, y, aggregation),
        matrices=matrices,
        notice=notice,
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
        If the run, the loading aggregation, the heatmap values, or the
        segment format is invalid.
    """
    if request.aggregation is not None and request.aggregation not in LOADING_AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {request.aggregation!r}")
    view = request.view
    if view is not None and view != COMPONENT_VIEW and not is_parameter_key(view):
        raise ValueError(f"unknown view: {request.view!r}")
    parse_segment(request.segment)
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    by_id = {str(run["run_id"]): run for run in runs}
    if request.run is not None and request.run not in by_id:
        raise ValueError(f"succeeded fit run not found: {request.run}")
    run = by_id[request.run] if request.run is not None else (runs[0] if runs else None)
    if run is None:
        return ModelView(runs=runs, error="No succeeded fit run.")
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return ModelView(runs=runs, run_id=str(run["run_id"]), error=str(error))
    return _run_view(run, artifacts, cache, request, runs)


def heatmap_request(view: ModelView) -> HeatmapRequest | None:
    """Return the key of the heatmap matrix a resolved view draws.

    Parameters
    ----------
    view : ModelView
        Resolved view.

    Returns
    -------
    HeatmapRequest | None
        Run, values, component, and segment of the drawn matrix, or
        ``None`` when the view draws none.
    """
    segment = format_segment(view.segment)
    if not view.matrices or view.run_id is None or segment is None:
        return None
    return HeatmapRequest(
        run=view.run_id, view=view.view, component=view.component, segment=segment
    )
