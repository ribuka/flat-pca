"""Choice of the data shown on the T² and Q screen."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig
from flat_pca.utils import natural_keys

from .display_cache import DisplayCache
from .fit_artifacts import DisplayArtifacts, RunArtifactError, artifact_error
from .scored_samples import choose_metadata_column, metadata_columns, scored_samples

_ROW = "row"
_STEM_RANK = "stem_rank"
_ORDER_VALUE = "order_value"


@dataclass(frozen=True)
class MonitoringRequest:
    """Choices requested by the T² and Q screen's query parameters.

    Attributes
    ----------
    run : str | None
        Fit run shown; ``None`` for the latest succeeded fit run.
    order : str | None
        Metadata column ordering the control charts; ``None`` for the
        default column and ``""`` for the natural order of the stems.
    """

    run: str | None = None
    order: str | None = None


@dataclass(frozen=True)
class MonitoringPoints:
    """T² and Q of the scored files, in control-chart order.

    Attributes
    ----------
    samples : pl.DataFrame
        ``stem`` and the metadata columns of each scored file. Files
        dropped by the imputation have no score and are absent.
    t2 : np.ndarray
        ``float64`` squared Mahalanobis distance (T²) of each file.
    q : np.ndarray
        ``float64`` Q statistic (SPE) of each file.
    t2_ucl : float
        Upper control limit of T².
    q_ucl : float
        Upper control limit of Q.
    """

    samples: pl.DataFrame
    t2: np.ndarray
    q: np.ndarray
    t2_ucl: float
    q_ucl: float

    @property
    def t2_exceeding(self) -> list[str]:
        """Return the stems whose T² exceeds its UCL, in display order."""
        return self.samples["stem"].filter(self.t2 > self.t2_ucl).to_list()

    @property
    def q_exceeding(self) -> list[str]:
        """Return the stems whose Q exceeds its UCL, in display order."""
        return self.samples["stem"].filter(self.q > self.q_ucl).to_list()


@dataclass(frozen=True)
class MonitoringView:
    """Resolved choices and the data of the T² and Q screen.

    Attributes
    ----------
    runs : list[dict[str, object]]
        Succeeded fit runs, newest first.
    run_id : str | None
        Fit run in use, or ``None`` when no fit run succeeded.
    order_options : list[str]
        Metadata columns of the run's samples.
    order : str | None
        Chosen ordering column, or ``None`` for the natural order of the
        stems.
    points : MonitoringPoints | None
        T² and Q of the scored files in the chosen order.
    unscored : list[str]
        Stems of the run without a score, dropped by the imputation.
    error : str | None
        Message shown instead of the screen.
    """

    runs: list[dict[str, object]] = field(default_factory=list)
    run_id: str | None = None
    order_options: list[str] = field(default_factory=list)
    order: str | None = None
    points: MonitoringPoints | None = None
    unscored: list[str] = field(default_factory=list)
    error: str | None = None


def control_chart_order(samples: pl.DataFrame, order_by: str | None) -> np.ndarray:
    """Return the row order of the files in the control charts.

    Parameters
    ----------
    samples : pl.DataFrame
        Frame with ``stem`` and the metadata columns.
    order_by : str | None
        Metadata column sorted in ascending order with missing values last;
        ties and ``None`` fall back to the natural order of the stems.

    Returns
    -------
    np.ndarray
        Zero-based row positions of ``samples`` in display order.
    """
    stems = samples["stem"].to_list()
    by_stem = sorted(range(len(stems)), key=lambda row: natural_keys(str(stems[row])))
    ranks = np.empty(len(stems), dtype=np.int64)
    ranks[by_stem] = np.arange(len(stems))
    # Only internal names, so no metadata column name can collide with them.
    frame = pl.DataFrame({_ROW: np.arange(len(stems)), _STEM_RANK: ranks})
    keys = [_STEM_RANK]
    if order_by is not None:
        frame = frame.with_columns(samples[order_by].alias(_ORDER_VALUE))
        keys = [_ORDER_VALUE, _STEM_RANK]
    return frame.sort(keys, nulls_last=True)[_ROW].to_numpy().astype(np.intp)


def statistic_configs(run: dict[str, object]) -> tuple[MahalanobisConfig, SpeConfig]:
    """Return the T² and Q settings of a fit run.

    Parameters
    ----------
    run : dict[str, object]
        The fit run's ``runs`` row.

    Returns
    -------
    tuple[MahalanobisConfig, SpeConfig]
        Settings saved in the run's configuration.

    Raises
    ------
    RunArtifactError
        If the configuration lacks or holds invalid settings.
    """
    try:
        config = cast(dict[str, object], json.loads(str(run["config_json"])))
        return (
            MahalanobisConfig(**cast(dict[str, object], config["mahalanobis"])),  # type: ignore[arg-type]
            SpeConfig(**cast(dict[str, object], config["spe"])),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise artifact_error(error) from error


def monitoring_points(
    artifacts: DisplayArtifacts,
    mahalanobis: MahalanobisConfig,
    spe: SpeConfig,
    order_by: str | None,
) -> MonitoringPoints:
    """Return T² and Q of the scored files in control-chart order.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        The run's artifacts.
    mahalanobis : MahalanobisConfig
        T² settings of the run, naming the ``scores.parquet`` columns.
    spe : SpeConfig
        Q settings of the run, naming the ``scores.parquet`` columns.
    order_by : str | None
        Ordering column; see ``control_chart_order``.

    Returns
    -------
    MonitoringPoints
        Statistics and control limits of the scored files.

    Raises
    ------
    RunArtifactError
        If ``scores.parquet`` lacks a statistic column or holds no row.
    """
    wanted = [
        mahalanobis.distance_column,
        mahalanobis.ucl_column,
        spe.spe_column,
        spe.ucl_column,
    ]
    missing = [column for column in wanted if column not in artifacts.scores.columns]
    if missing:
        raise artifact_error(ValueError(f"scores.parquet lacks columns {missing}"))
    if artifacts.scores.height == 0:
        raise artifact_error(ValueError("scores.parquet holds no row"))
    samples, scores = scored_samples(artifacts)
    order = control_chart_order(samples, order_by)
    return MonitoringPoints(
        samples=samples[order],
        t2=scores[mahalanobis.distance_column].cast(pl.Float64).to_numpy()[order],
        q=scores[spe.spe_column].cast(pl.Float64).to_numpy()[order],
        t2_ucl=float(scores[mahalanobis.ucl_column][0]),
        q_ucl=float(scores[spe.ucl_column][0]),
    )


def resolve_monitoring(
    cache: DisplayCache,
    fit_runs: list[dict[str, object]],
    request: MonitoringRequest,
    default_order: str | None,
) -> MonitoringView:
    """Resolve the requested choices and load the data they show.

    Parameters
    ----------
    cache : DisplayCache
        Display cache of the workspace.
    fit_runs : list[dict[str, object]]
        Fit runs, newest first; only succeeded ones are used.
    request : MonitoringRequest
        Requested choices.
    default_order : str | None
        Default ordering column (``ui.default_order_by``).

    Returns
    -------
    MonitoringView
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
        return MonitoringView(runs=runs, error="成功した fit run がありません")
    run_id = str(run["run_id"])
    try:
        artifacts = cache.fit_artifacts(Path(str(run["artifact_dir"])))
    except RunArtifactError as error:
        return MonitoringView(runs=runs, run_id=run_id, error=str(error))
    order_options = metadata_columns(artifacts.samples)
    order = choose_metadata_column(request.order, default_order, order_options)
    common = {
        "runs": runs,
        "run_id": run_id,
        "order_options": order_options,
        "order": order,
    }
    try:
        points = monitoring_points(artifacts, *statistic_configs(run), order)
    except RunArtifactError as error:
        return MonitoringView(**common, error=str(error))  # type: ignore[arg-type]
    scored = set(points.samples["stem"].to_list())
    return MonitoringView(
        **common,  # type: ignore[arg-type]
        points=points,
        unscored=[stem for stem in artifacts.samples["stem"].to_list() if stem not in scored],
    )
