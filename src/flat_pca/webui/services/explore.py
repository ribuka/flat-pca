"""Choice of the data shown on the spectral exploration screen."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl

from flat_pca.feature_engineering.pca import PcaModel, PreparedRows, q_contribution
from flat_pca.feature_engineering.pca.mahalanobis import resolve_mahalanobis_components

from ..database import Database
from .catalog_query import FileQuery, list_files, list_segments
from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError
from .monitoring import statistic_configs
from .q_consistency import QMismatch, q_mismatch, saved_q_by_stem
from .reconstruction import ReconstructionKind, reconstruction_values
from .spectral_matrix import (
    SpectralMatrix,
    TrendLine,
    feature_segment_matrix,
    raw_segment_matrix,
    trend_at_step_time,
    trend_at_wavelength,
)

ExploreViewKind = Literal[
    "raw",
    "preprocessed",
    "component",
    "contribution",
    "reconstruction",
    "residual",
    "q_contribution",
]
# Views computed from the rows prepared with the run's imputation.
PreparedViewKind = Literal["contribution", "reconstruction", "residual", "q_contribution"]
VIEW_LABELS: dict[ExploreViewKind, str] = {
    "raw": "元データ",
    "preprocessed": "前処理済み",
    "component": "PCA 成分",
    "contribution": "第 k 成分のみの寄与",
    "reconstruction": "累積再構成（1..k 成分）",
    "residual": "残差",
    "q_contribution": "Q 寄与",
}
VALUE_NAMES: dict[ExploreViewKind, str] = {
    "component": "coefficient",
    "contribution": "contribution",
    "residual": "residual",
    "q_contribution": "q_contribution",
}


@dataclass(frozen=True)
class ExploreRequest:
    """Choices requested by the exploration screen's query parameters.

    Attributes
    ----------
    view : str
        A key of ``VIEW_LABELS``.
    run : str | None
        Fit run whose artifacts are shown; ``None`` for the latest
        succeeded fit run.
    files : tuple[str, ...]
        Stems to show. The first shown one is drawn as the heatmap and all
        of them are overlaid in the trends.
    segment : str | None
        ``"{Step}:{Sequence}"`` to show.
    component : int | None
        1-based component number k of the component and reconstruction
        views.
    """

    view: str = "raw"
    run: str | None = None
    files: tuple[str, ...] = ()
    segment: str | None = None
    component: int | None = None


@dataclass(frozen=True)
class ExploreView:
    """Resolved choices and the matrices of one exploration view.

    Attributes
    ----------
    view : ExploreViewKind
        Shown data kind.
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.
    run_id : str | None
        Fit run in use, or ``None`` when no fit run succeeded.
    file_options : list[str]
        Stems that can be chosen; empty for the component view.
    files : list[str]
        Chosen stems in option order.
    segment_options : list[tuple[int, int]]
        ``(Step, Sequence)`` pairs that can be chosen.
    segment : tuple[int, int] | None
        Chosen pair.
    component_count : int
        Number of components of the run, or 0 without a run.
    component : int | None
        Chosen 1-based component number k of the component and
        reconstruction views.
    q_components : int | None
        Number of leading components the Q contribution view reconstructs
        from, fixed by the run's ``SpeConfig``.
    value_name : str
        ``"coefficient"`` for components, ``"contribution"``,
        ``"residual"``, and ``"q_contribution"`` for those views, otherwise
        ``"intensity"``.
    intensity_transform : dict[str, object] | None
        ``name`` and ``scale`` of the run's intensity transform, given for
        the preprocessed and reconstruction views.
    matrices : dict[str, SpectralMatrix]
        Unbinned matrices keyed by trace label; the first is the heatmap.
    skipped : list[str]
        Chosen stems lacking the chosen segment.
    dropped : list[str]
        Chosen stems whose rows the run's ``impute_strategy="drop"``
        drops, so they cannot be reconstructed.
    q_mismatches : list[QMismatch]
        Chosen stems of the Q contribution view whose unbinned
        contributions do not add up to their saved Q.
    error : str | None
        Message shown instead of the view.
    """

    view: ExploreViewKind
    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    segment_options: list[tuple[int, int]] = field(default_factory=list)
    segment: tuple[int, int] | None = None
    component_count: int = 0
    component: int | None = None
    q_components: int | None = None
    value_name: str = "intensity"
    intensity_transform: dict[str, object] | None = None
    matrices: dict[str, SpectralMatrix] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    q_mismatches: list[QMismatch] = field(default_factory=list)
    error: str | None = None


def parse_segment(text: str | None) -> tuple[int, int] | None:
    """Parse ``"{Step}:{Sequence}"``.

    Parameters
    ----------
    text : str | None
        Query parameter value.

    Returns
    -------
    tuple[int, int] | None
        The pair, or ``None`` for an empty value.

    Raises
    ------
    ValueError
        If the value is not two integers separated by ``:``.
    """
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"segment must be '<Step>:<Sequence>': {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise ValueError(f"segment must be '<Step>:<Sequence>': {text!r}") from error


def _choose_files(options: list[str], requested: Sequence[str]) -> list[str]:
    """Return the requested options, or the first option without any.

    Parameters
    ----------
    options : list[str]
        Stems that can be chosen.
    requested : Sequence[str]
        Requested stems; unknown ones are ignored.

    Returns
    -------
    list[str]
        Chosen stems in option order.
    """
    wanted = set(requested)
    chosen = [stem for stem in options if stem in wanted]
    return chosen or options[:1]


def _choose_segment(
    options: list[tuple[int, int]], requested: tuple[int, int] | None
) -> tuple[int, int] | None:
    """Return the requested segment if available, otherwise the first one.

    Parameters
    ----------
    options : list[tuple[int, int]]
        Available pairs.
    requested : tuple[int, int] | None
        Requested pair.

    Returns
    -------
    tuple[int, int] | None
        Chosen pair, or ``None`` without options.
    """
    if requested in options:
        return requested
    return options[0] if options else None


def _feature_segments(artifacts: DisplayArtifacts) -> list[tuple[int, int]]:
    """Return the ``(Step, Sequence)`` pairs of a run's features.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        Fit-run artifacts.

    Returns
    -------
    list[tuple[int, int]]
        Pairs in ascending order.
    """
    pairs = artifacts.features.select("Step", "Sequence").unique().sort("Step", "Sequence")
    return [(int(step), int(sequence)) for step, sequence in pairs.iter_rows()]


def _raw_view(
    database: Database,
    selected: list[str],
    cache: DisplayCache,
    request: ExploreRequest,
    runs: list[dict[str, object]],
    run_id: str | None,
) -> ExploreView:
    """Resolve the raw-data view.

    Parameters
    ----------
    database : Database
        Workspace database.
    selected : list[str]
        Stems selected on the data selection screen. Without any, every
        cataloged file can be chosen.
    cache : DisplayCache
        Display cache holding the read files.
    request : ExploreRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.
    run_id : str | None
        Fit run in use.

    Returns
    -------
    ExploreView
        Raw spectra of the chosen files.
    """
    paths = {
        str(file["stem"]): Path(str(file["path"]))
        for file in list_files(database, FileQuery())
    }
    options = [stem for stem in selected if stem in paths] or list(paths)
    if not options:
        return ExploreView(
            view="raw", runs=runs, run_id=run_id, error="catalog にファイルがありません"
        )
    files = _choose_files(options, request.files)
    segment_options = list_segments(database, files[0])
    segment = _choose_segment(segment_options, parse_segment(request.segment))
    matrices: dict[str, SpectralMatrix] = {}
    skipped: list[str] = []
    error = None
    for stem in files if segment is not None else []:
        try:
            spectra = cache.raw_spectra(paths[stem])
        except (OSError, ValueError, pl.exceptions.PolarsError) as caught:
            error = f"{stem} を読み込めません（{type(caught).__name__}: {caught}）"
            break
        try:
            matrices[stem] = raw_segment_matrix(spectra, *segment)  # type: ignore[misc]
        except ValueError:
            skipped.append(stem)
    return ExploreView(
        view="raw",
        runs=runs,
        run_id=run_id,
        file_options=options,
        files=files,
        segment_options=segment_options,
        segment=segment,
        matrices=matrices if error is None else {},
        skipped=skipped,
        error=error,
    )


def _choose_component(requested: int | None, component_count: int) -> int:
    """Return the requested component number if available, otherwise 1.

    Parameters
    ----------
    requested : int | None
        Requested 1-based component number.
    component_count : int
        Number of components of the run.

    Returns
    -------
    int
        Chosen 1-based component number.
    """
    if requested is None or not 1 <= requested <= component_count:
        return 1
    return requested


def _reconstruction_matrices(
    view: PreparedViewKind,
    run_dir: Path,
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    files: list[str],
    segment: tuple[int, int],
    values_of: Callable[[str, PcaModel, PreparedRows], np.ndarray],
) -> tuple[dict[str, SpectralMatrix], list[str]]:
    """Compute the matrices of the views of prepared rows for the chosen files.

    Parameters
    ----------
    view : PreparedViewKind
        ``"contribution"``, ``"reconstruction"``, ``"residual"``, or
        ``"q_contribution"``.
    run_dir : Path
        Run directory of the fit run.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    files : list[str]
        Chosen stems.
    segment : tuple[int, int]
        Chosen ``(Step, Sequence)``.
    values_of : Callable[[str, PcaModel, PreparedRows], np.ndarray]
        Feature values shown for one kept prepared row, given its stem.

    Returns
    -------
    tuple[dict[str, SpectralMatrix], list[str]]
        Matrices keyed by trace label, and the stems whose rows the
        imputation drops. The reconstruction view also holds each file's
        preprocessed values, so the trends overlay the two.

    Raises
    ------
    RunArtifactError
        If the model cannot be read.
    """
    model = cache.pca_model(run_dir)
    stems = artifacts.samples["stem"].to_list()
    matrices: dict[str, SpectralMatrix] = {}
    dropped: list[str] = []
    for stem in files:
        row = stems.index(stem)
        prepared = cache.prepared_row(run_dir, row)
        if prepared.kept.size == 0:
            dropped.append(stem)
            continue
        values = values_of(stem, model, prepared)
        if view != "reconstruction":
            matrices[stem] = feature_segment_matrix(artifacts.features, values, *segment)
            continue
        matrices[f"{stem}（累積再構成）"] = feature_segment_matrix(
            artifacts.features, values, *segment
        )
        matrices[f"{stem}（前処理済み）"] = feature_segment_matrix(
            artifacts.features,
            # Read only this row of the memory-mapped matrix.
            np.asarray(artifacts.x[row], dtype=np.float64),
            *segment,
        )
    return matrices, dropped


def _run_view(
    view: ExploreViewKind,
    run: dict[str, object],
    artifacts: DisplayArtifacts,
    cache: DisplayCache,
    request: ExploreRequest,
    runs: list[dict[str, object]],
) -> ExploreView:
    """Resolve a view of one fit run's artifacts.

    Parameters
    ----------
    view : ExploreViewKind
        Any view but ``"raw"``.
    run : dict[str, object]
        The fit run's ``runs`` row.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    request : ExploreRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.

    Returns
    -------
    ExploreView
        Reshaped ``X.npy`` rows, one reshaped component, or the
        contributions, reconstructions, residuals, or Q contributions of the
        chosen files.
    """
    segment_options = _feature_segments(artifacts)
    segment = _choose_segment(segment_options, parse_segment(request.segment))
    assert segment is not None  # a fit run always has features
    component_count = artifacts.components.shape[0]
    component = _choose_component(request.component, component_count)
    common = {
        "runs": runs,
        "run_id": str(run["run_id"]),
        "segment_options": segment_options,
        "segment": segment,
        "component_count": component_count,
        "value_name": VALUE_NAMES.get(view, "intensity"),
    }
    if view == "component":
        # Read only this row of the memory-mapped components.
        values = np.asarray(artifacts.components[component - 1], dtype=np.float64)
        return ExploreView(
            view=view,
            **common,  # type: ignore[arg-type]
            component=component,
            matrices={
                f"PC{component}": feature_segment_matrix(artifacts.features, values, *segment)
            },
        )
    stems = artifacts.samples["stem"].to_list()
    files = _choose_files(stems, request.files)
    preprocess = cast(dict[str, object], json.loads(str(run["config_json"]))["preprocess"])
    common |= {
        "file_options": stems,
        "files": files,
        "intensity_transform": {
            "name": preprocess.get("intensity_transform", "none"),
            "scale": preprocess.get("intensity_transform_scale", 1.0),
        },
    }
    if view == "preprocessed":
        matrices = {
            stem: feature_segment_matrix(
                artifacts.features,
                # Read only this row of the memory-mapped matrix.
                np.asarray(artifacts.x[stems.index(stem)], dtype=np.float64),
                *segment,
            )
            for stem in files
        }
        return ExploreView(view=view, **common, matrices=matrices)  # type: ignore[arg-type]
    run_dir = Path(str(run["artifact_dir"]))
    shown_component: int | None = component
    q_components: int | None = None
    q_mismatches: list[QMismatch] = []
    try:
        if view == "q_contribution":
            # Q fixes its own component count; the chosen k is not used.
            shown_component = None
            spe = statistic_configs(run)[1]
            selector = spe.cumulative_explained_variance
            q_components = resolve_mahalanobis_components(
                cache.pca_model(run_dir).pca, selector
            )
            saved_q = saved_q_by_stem(artifacts, spe.spe_column)

            def values_of(stem: str, model: PcaModel, prepared: PreparedRows) -> np.ndarray:
                """Return the Q contribution of the prepared row.

                Its sum over every feature is compared with the saved Q,
                which float32 artifacts may fail to reproduce.
                """
                values = q_contribution(prepared, model, selector)[0]
                if stem in saved_q:
                    mismatch = q_mismatch(stem, float(values.sum()), saved_q[stem])
                    if mismatch is not None:
                        q_mismatches.append(mismatch)
                return values

        else:

            def values_of(stem: str, model: PcaModel, prepared: PreparedRows) -> np.ndarray:
                """Return the reconstruction-view values of the prepared row."""
                return reconstruction_values(
                    cast(ReconstructionKind, view), model, prepared, component
                )

        matrices, dropped = _reconstruction_matrices(
            cast(PreparedViewKind, view), run_dir, artifacts, cache, files, segment, values_of
        )
    except RunArtifactError as error:
        return ExploreView(
            view=view,
            **common,  # type: ignore[arg-type]
            component=shown_component,
            error=str(error),
        )
    error = None
    if not matrices:
        error = "選んだファイルはすべて欠損値を含み、補完方法 drop で除外されるため表示できません"
    return ExploreView(
        view=view,
        **common,  # type: ignore[arg-type]
        component=shown_component,
        q_components=q_components,
        matrices=matrices,
        dropped=dropped,
        q_mismatches=q_mismatches,
        error=error,
    )




def resolve_explore(
    database: Database,
    selected: list[str],
    cache: DisplayCache,
    fit_runs: list[dict[str, object]],
    request: ExploreRequest,
) -> ExploreView:
    """Resolve the requested choices and load the matrices they show.

    Unavailable choices fall back to the first available one, so a view is
    shown whenever data exist.

    Parameters
    ----------
    database : Database
        Workspace database.
    selected : list[str]
        Stems selected on the data selection screen.
    cache : DisplayCache
        Display cache of the workspace.
    fit_runs : list[dict[str, object]]
        Fit runs, newest first; only succeeded ones are used.
    request : ExploreRequest
        Requested choices.

    Returns
    -------
    ExploreView
        Resolved choices and matrices, or an error message.

    Raises
    ------
    ValueError
        If the view kind, segment format, or run is invalid.
    """
    if request.view not in VIEW_LABELS:
        raise ValueError(f"unknown view: {request.view!r}")
    view = cast(ExploreViewKind, request.view)
    parse_segment(request.segment)
    runs = [run for run in fit_runs if run["status"] == "succeeded"]
    by_id = {str(run["run_id"]): run for run in runs}
    if request.run is not None and request.run not in by_id:
        raise ValueError(f"succeeded fit run not found: {request.run}")
    run = by_id[request.run] if request.run is not None else (runs[0] if runs else None)
    run_id = None if run is None else str(run["run_id"])
    if view == "raw":
        return _raw_view(database, selected, cache, request, runs, run_id)
    if run is None:
        return ExploreView(view=view, runs=runs, error="成功した fit run がありません")
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return ExploreView(view=view, runs=runs, run_id=run_id, error=str(error))
    return _run_view(view, run, artifacts, cache, request, runs)


def shown_request(view: ExploreView) -> ExploreRequest:
    """Return the request that selects exactly the resolved choices of a view.

    Parameters
    ----------
    view : ExploreView
        Resolved view.

    Returns
    -------
    ExploreRequest
        Request with the view's run, files, segment, and component, so
        requests built from the page's trend query equal it.
    """
    return ExploreRequest(
        view=view.view,
        run=view.run_id,
        files=tuple(view.files),
        segment=None if view.segment is None else f"{view.segment[0]}:{view.segment[1]}",
        component=view.component,
    )


def explore_trends(
    matrices: dict[str, SpectralMatrix], wavelength: float, step_time: float
) -> tuple[dict[str, TrendLine], dict[str, TrendLine]]:
    """Cut every matrix of a view at the nearest grid point.

    Parameters
    ----------
    matrices : dict[str, SpectralMatrix]
        Unbinned matrices of a view keyed by trace label.
    wavelength : float
        Requested wavelength.
    step_time : float
        Requested ``StepTime``.

    Returns
    -------
    tuple[dict[str, TrendLine], dict[str, TrendLine]]
        Lines over ``StepTime`` at ``wavelength``, and lines over
        wavelength at ``step_time``, keyed by trace label. Each matrix is
        cut at its own nearest grid point.
    """
    return (
        {label: trend_at_wavelength(matrix, wavelength) for label, matrix in matrices.items()},
        {label: trend_at_step_time(matrix, step_time) for label, matrix in matrices.items()},
    )
