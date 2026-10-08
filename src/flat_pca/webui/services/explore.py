"""Choice of the data shown on the spectral exploration screen."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl

from flat_pca.feature_engineering.pca import PcaModel, PreparedRows, q_contribution
from flat_pca.feature_engineering.pca.mahalanobis import resolve_mahalanobis_components
from flat_pca.spectral.schema import SOURCE_COLUMN
from flat_pca.utils import natural_keys

from ..run_config import run_config, statistic_configs
from .component_choice import choose_component
from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError
from .q_consistency import QMismatch, q_mismatch, saved_q_by_stem
from .reconstruction import ReconstructionKind, reconstruction_values
from .run_choice import choose_run
from .run_dirs import RunDirs, run_dirs
from .segment_choice import (
    choose_segment,
    feature_segments,
    format_segment,
    parse_segment,
    raw_segments,
)
from .spectral_matrix import (
    SpectralMatrix,
    TrendLine,
    feature_segment_matrix,
    raw_segment_matrix,
    trend_at_step_time,
    trend_at_wavelength,
)
from .view_selection import NO_TRANSFORM_RUN

ExploreViewKind = Literal[
    "raw",
    "preprocessed",
    "contribution",
    "reconstruction",
    "residual",
    "q_contribution",
]
# Views computed from the rows prepared with the run's imputation.
PreparedViewKind = Literal["contribution", "reconstruction", "residual", "q_contribution"]
VIEW_LABELS: dict[ExploreViewKind, str] = {
    "raw": "Raw",
    "preprocessed": "Preprocessed",
    "contribution": "Contribution of component k",
    "reconstruction": "Reconstruction (components 1..k)",
    "residual": "Residual",
    "q_contribution": "Q contribution",
}
VALUE_NAMES: dict[ExploreViewKind, str] = {
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
        Transform run whose artifacts are shown; ``None`` for the latest
        succeeded transform run.
    files : tuple[str, ...]
        Stems to show, all of them overlaid in the trends.
    heatmap_file : str | None
        Shown stem drawn as the heatmap; the first shown one by default.
    segment : str | None
        ``"{Step}:{Sequence}"`` to show.
    component : int | None
        1-based component number k of the contribution, reconstruction,
        and residual views.
    """

    view: str = "raw"
    run: str | None = None
    files: tuple[str, ...] = ()
    heatmap_file: str | None = None
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
        Succeeded transform runs to choose from, newest first.
    run_id : str | None
        Transform run in use, or ``None`` when no transform run succeeded.
    file_options : list[str]
        Transform targets of the run that can be chosen, in natural order.
    files : list[str]
        Chosen stems in option order.
    heatmap_options : list[str]
        Chosen stems that have matrices, in option order; any of them can be
        drawn as the heatmap.
    heatmap_file : str | None
        Stem drawn as the heatmap, or ``None`` when nothing is shown.
    segment_options : list[tuple[int, int]]
        ``(Step, Sequence)`` pairs that can be chosen.
    segment : tuple[int, int] | None
        Chosen pair.
    component_count : int
        Number of components of the run, or 0 without a run.
    component : int | None
        Chosen 1-based component number k of the contribution,
        reconstruction, and residual views.
    q_components : int | None
        Number of leading components the Q contribution view reconstructs
        from, fixed by the run's ``SpeConfig``.
    value_name : str
        ``"contribution"``, ``"residual"``, and ``"q_contribution"`` for
        those views, otherwise ``"intensity"``.
    intensity_transform : dict[str, object] | None
        ``name`` and ``scale`` of the run's intensity transform, given for
        the preprocessed and reconstruction views.
    matrices : dict[str, SpectralMatrix]
        Unbinned matrices keyed by trace label. The traces of the heatmap
        file come first, followed by those of the other files in option
        order; the first trace is the heatmap.
    skipped : list[str]
        Chosen stems lacking the chosen segment.
    dropped : list[str]
        Chosen stems whose rows the run's ``impute_strategy="drop"``
        drops, so they cannot be reconstructed.
    q_mismatches : list[QMismatch]
        Chosen stems of the Q contribution view whose unbinned
        contributions do not add up to their saved Q.
    omitted : list[str]
        Requested stems left out beyond ``ui.explore_max_files``.
    error : str | None
        Message shown instead of the view.
    """

    view: ExploreViewKind
    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    file_options: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    heatmap_options: list[str] = field(default_factory=list)
    heatmap_file: str | None = None
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
    omitted: list[str] = field(default_factory=list)
    error: str | None = None


def _choose_files(
    options: list[str], requested: Sequence[str], max_files: int
) -> tuple[list[str], list[str]]:
    """Return the requested options up to a limit, or the first option without any.

    Parameters
    ----------
    options : list[str]
        Stems that can be chosen.
    requested : Sequence[str]
        Requested stems; unknown ones are ignored.
    max_files : int
        Maximum number of chosen stems.

    Returns
    -------
    tuple[list[str], list[str]]
        Chosen stems in option order, at most ``max_files`` of them, and
        the requested stems left out beyond the limit.
    """
    wanted = set(requested)
    chosen = [stem for stem in options if stem in wanted] or options[:1]
    return chosen[:max_files], chosen[max_files:]


def _heatmap_first(files: list[str], requested: str | None) -> list[str]:
    """Return the chosen stems with the requested heatmap file moved to the front.

    Parameters
    ----------
    files : list[str]
        Chosen stems in option order.
    requested : str | None
        Requested heatmap file; ignored unless it is chosen.

    Returns
    -------
    list[str]
        Stems in the order their matrices are built, so the first shown one
        is drawn as the heatmap.
    """
    if requested not in files:
        return files
    return [requested, *(stem for stem in files if stem != requested)]


def _heatmap_file(shown: list[str], requested: str | None) -> str | None:
    """Return the stem drawn as the heatmap.

    Parameters
    ----------
    shown : list[str]
        Chosen stems that have matrices, in option order.
    requested : str | None
        Requested heatmap file.

    Returns
    -------
    str | None
        The requested stem if it is shown, or else the first shown one
        (``None`` without any).
    """
    if requested in shown:
        return requested
    return shown[0] if shown else None


def _raw_view(
    cache: DisplayCache,
    request: ExploreRequest,
    runs: list[dict[str, object]],
    run_id: str,
    artifacts: DisplayArtifacts,
    max_files: int,
) -> ExploreView:
    """Resolve the raw-data view of the run's transform targets.

    Parameters
    ----------
    cache : DisplayCache
        Display cache holding the read files.
    request : ExploreRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded transform runs to choose from, newest first.
    run_id : str
        Fit or transform run in use.
    artifacts : DisplayArtifacts
        The run's artifacts, whose ``samples.parquet`` lists the transform
        targets and their paths (``source``).
    max_files : int
        Maximum number of shown files.

    Returns
    -------
    ExploreView
        Raw spectra of the chosen files. The ``(Step, Sequence)`` options
        are those of the requested heatmap file, or else the first chosen
        file.
    """
    stems = artifacts.samples["stem"].to_list()
    paths = dict(zip(stems, artifacts.samples[SOURCE_COLUMN].to_list(), strict=True))
    options = sorted(stems, key=natural_keys)
    files, omitted = _choose_files(options, request.files, max_files)
    order = _heatmap_first(files, request.heatmap_file)
    segment_options: list[tuple[int, int]] = []
    segment: tuple[int, int] | None = None
    matrices: dict[str, SpectralMatrix] = {}
    skipped: list[str] = []
    error = None
    for stem in order:
        try:
            spectra = cache.raw_spectra(Path(str(paths[stem])))
        except (OSError, ValueError, pl.exceptions.PolarsError) as caught:
            error = f"Cannot read {stem} ({type(caught).__name__}: {caught})."
            break
        if stem == order[0]:
            segment_options = raw_segments(spectra)
            segment = choose_segment(segment_options, parse_segment(request.segment))
        if segment is None:
            break
        try:
            matrices[stem] = raw_segment_matrix(spectra, *segment)
        except ValueError:
            skipped.append(stem)
    if error is not None:
        matrices = {}
    shown = [stem for stem in files if stem in matrices]
    return ExploreView(
        view="raw",
        runs=runs,
        run_id=run_id,
        file_options=options,
        files=files,
        heatmap_options=shown,
        heatmap_file=_heatmap_file(shown, request.heatmap_file),
        segment_options=segment_options,
        segment=segment,
        matrices=matrices,
        skipped=[stem for stem in files if stem in skipped],
        omitted=omitted,
        error=error,
    )


def _reconstruction_matrices(
    view: PreparedViewKind,
    dirs: RunDirs,
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
    dirs : RunDirs
        Directories of the run's model and data.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    files : list[str]
        Chosen stems, in the order their matrices are built.
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
    model = cache.pca_model(dirs.model)
    stems = artifacts.samples["stem"].to_list()
    matrices: dict[str, SpectralMatrix] = {}
    dropped: list[str] = []
    for stem in files:
        row = stems.index(stem)
        prepared = cache.prepared_row(dirs, row)
        if prepared.kept.size == 0:
            dropped.append(stem)
            continue
        values = values_of(stem, model, prepared)
        if view != "reconstruction":
            matrices[stem] = feature_segment_matrix(artifacts.features, values, *segment)
            continue
        matrices[f"{stem} (reconstruction)"] = feature_segment_matrix(
            artifacts.features, values, *segment
        )
        matrices[f"{stem} (preprocessed)"] = feature_segment_matrix(
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
    max_files: int,
) -> ExploreView:
    """Resolve a view of one transform run's artifacts.

    Parameters
    ----------
    view : ExploreViewKind
        Any view but ``"raw"``.
    run : dict[str, object]
        The fit or transform run's ``runs`` row.
    artifacts : DisplayArtifacts
        The run's artifacts.
    cache : DisplayCache
        Display cache holding the model and the prepared rows.
    request : ExploreRequest
        Requested choices.
    runs : list[dict[str, object]]
        Succeeded transform runs to choose from, newest first.
    max_files : int
        Maximum number of shown files.

    Returns
    -------
    ExploreView
        Reshaped ``X.npy`` rows, or the
        contributions, reconstructions, residuals, or Q contributions of the
        chosen files.
    """
    segment_options = feature_segments(artifacts)
    segment = choose_segment(segment_options, parse_segment(request.segment))
    assert segment is not None  # a fit run always has features
    component_count = artifacts.components.shape[0]
    component = choose_component(request.component, 1, component_count)
    stems = artifacts.samples["stem"].to_list()
    options = sorted(stems, key=natural_keys)
    files, omitted = _choose_files(options, request.files, max_files)
    order = _heatmap_first(files, request.heatmap_file)
    config = run_config(run)
    preprocess = config["preprocess"]
    base = ExploreView(
        view=view,
        runs=runs,
        run_id=str(run["run_id"]),
        file_options=options,
        files=files,
        segment_options=segment_options,
        segment=segment,
        component_count=component_count,
        value_name=VALUE_NAMES.get(view, "intensity"),
        intensity_transform={
            "name": preprocess["intensity_transform"],
            "scale": preprocess["intensity_transform_scale"],
        },
        omitted=omitted,
    )
    if view == "preprocessed":
        matrices = {
            stem: feature_segment_matrix(
                artifacts.features,
                # Read only this row of the memory-mapped matrix.
                np.asarray(artifacts.x[stems.index(stem)], dtype=np.float64),
                *segment,
            )
            for stem in order
        }
        return replace(
            base,
            heatmap_options=files,
            heatmap_file=_heatmap_file(files, request.heatmap_file),
            matrices=matrices,
        )
    dirs = run_dirs(run)
    shown_component: int | None = component
    q_components: int | None = None
    q_mismatches: list[QMismatch] = []
    try:
        if view == "q_contribution":
            # Q fixes its own component count; the chosen k is not used.
            shown_component = None
            spe = statistic_configs(config)[1]
            selector = spe.cumulative_explained_variance
            q_components = resolve_mahalanobis_components(
                cache.pca_model(dirs.model).pca, selector
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
            cast(PreparedViewKind, view), dirs, artifacts, cache, order, segment, values_of
        )
    except RunArtifactError as error:
        return replace(
            base,
            component=shown_component,
            error=str(error),
        )
    error = None
    if not matrices:
        error = "Cannot show the chosen files: all of them contain missing values and are dropped by the imputation strategy drop."
    shown = [stem for stem in files if stem not in dropped]
    return replace(
        base,
        heatmap_options=shown,
        heatmap_file=_heatmap_file(shown, request.heatmap_file),
        component=shown_component,
        q_components=q_components,
        matrices=matrices,
        dropped=[stem for stem in files if stem in dropped],
        q_mismatches=sorted(q_mismatches, key=lambda mismatch: files.index(mismatch.stem)),
        error=error,
    )


def resolve_explore(
    cache: DisplayCache,
    runs: list[dict[str, object]],
    request: ExploreRequest,
    max_files: int,
) -> ExploreView:
    """Resolve the requested choices and load the matrices they show.

    Unavailable choices fall back to the first available one, so a view is
    shown whenever data exist. Every view, the raw data included, chooses
    from the transform targets of the transform run (its ``samples.parquet``) in
    natural order.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    runs : list[dict[str, object]]
        Succeeded transform runs to choose from, newest first.
    request : ExploreRequest
        Requested choices.
    max_files : int
        Maximum number of shown files; requested files beyond it are left
        out and listed in ``omitted``.

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
    run = choose_run(runs, request.run, "transform")
    if run is None:
        return ExploreView(
            view=view,
            runs=runs,
            error=NO_TRANSFORM_RUN,
        )
    run_id = str(run["run_id"])
    try:
        artifacts = cache.display_artifacts(run_dirs(run))
    except RunArtifactError as error:
        return ExploreView(view=view, runs=runs, run_id=run_id, error=str(error))
    if view == "raw":
        return _raw_view(cache, request, runs, run_id, artifacts, max_files)
    return _run_view(view, run, artifacts, cache, request, runs, max_files)


def shown_request(view: ExploreView) -> ExploreRequest:
    """Return the request that selects exactly the resolved choices of a view.

    Parameters
    ----------
    view : ExploreView
        Resolved view.

    Returns
    -------
    ExploreRequest
        Request with the view's run, files, heatmap file, segment, and
        component, so requests built from the page's trend query equal it.
    """
    return ExploreRequest(
        view=view.view,
        run=view.run_id,
        files=tuple(view.files),
        heatmap_file=view.heatmap_file,
        segment=format_segment(view.segment),
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
