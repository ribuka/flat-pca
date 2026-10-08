"""Transform job: score files that were not fitted with a fit run's model.

The job runs in a child process and never touches the database. It
preprocesses and flattens the target files with the fit run's settings,
aligns them to the fit run's features, transforms them with its PCA
pipeline, and writes ``samples.parquet``, ``X.npy``, and ``scores.parquet``
to the transform run directory. The model and the features stay in the
fit run directory; the app process validates the results with
``services.transform_artifacts.register_transform_result``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from loguru import logger

from flat_pca.feature_engineering.flatten_pca import preprocess_and_flatten
from flat_pca.feature_engineering.preprocess.intensity_transform import (
    IntensityTransform,
    transform_intensity_values,
)
from flat_pca.spectral.schema import SOURCE_COLUMN, flattened_feature_columns

from ..services.fit_artifacts import load_pca_model
from ..services.run_dirs import FIT_RUN_DIR_KEY, FIT_RUN_ID_KEY
from ..settings import Settings
from .fit_run import (
    SAMPLES_FILE,
    SCORES_FILE,
    X_FILE,
    file_entries,
    sample_frame,
    score_frame,
)
from .progress import write_progress

STAGES = ("preprocess", "transform", "save")
# The fit run's features define the columns, so the targets' own grid and
# missing ratios prune none of them: downsampling keeps every value (it only
# selects grid points, after smoothing and normalization), and the inclusive
# bound of the missing ratio keeps every column. ``align_features`` then
# selects the fit run's features, and only then does the element-wise
# intensity transform run (``transform_target_intensity``), so values the fit
# run never used cannot fail its domain check.
TARGET_PREPROCESS_OVERRIDES: dict[str, object] = {
    "t_downsampling_stride": 1,
    "w_downsampling_stride": 1,
    "max_null_ratio": 1.0,
    "intensity_transform": "none",
}


def file_state(path: Path) -> dict[str, object]:
    """Return the size and modification time identifying a file's contents.

    Parameters
    ----------
    path : Path
        Target file.

    Returns
    -------
    dict[str, object]
        ``size`` and ``mtime_ns`` of the file, both ``None`` if it cannot be
        read (the job then fails on reading it).
    """
    try:
        stat = path.stat()
    except OSError:
        return {"size": None, "mtime_ns": None}
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_transform_config(
    settings: Settings,
    files: Sequence[Mapping[str, object]],
    fit_run: Mapping[str, object],
) -> dict[str, object]:
    """Build the JSON configuration of a transform job.

    Parameters
    ----------
    settings : Settings
        Application settings, giving the metadata columns and
        ``jobs.artifact_dtype``.
    files : Sequence[Mapping[str, object]]
        Target files with ``stem``, ``path``, and every metadata column.
    fit_run : Mapping[str, object]
        ``runs`` row of the succeeded fit run whose model transforms them.

    Returns
    -------
    dict[str, object]
        Configuration passed to ``run_transform`` and saved as
        ``config.json``: the fit run's identifier and directory, the
        target files (each with its ``file_state`` when the configuration
        is built), and the fit run's ``preprocess``, ``mahalanobis``, and
        ``spe`` settings.
    """
    fit_config = cast(dict[str, object], json.loads(str(fit_run["config_json"])))
    entries = file_entries(settings, files)
    for entry in cast(list[dict[str, object]], entries["files"]):
        entry |= file_state(Path(str(entry["path"])))
    return {
        FIT_RUN_ID_KEY: str(fit_run["run_id"]),
        FIT_RUN_DIR_KEY: str(fit_run["artifact_dir"]),
        **entries,
        "preprocess": fit_config["preprocess"],
        "mahalanobis": fit_config["mahalanobis"],
        "spe": fit_config["spe"],
        "artifact_dtype": settings.jobs.artifact_dtype,
    }


def align_features(flattened: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    """Select the fit run's features from flattened transform targets.

    Features of the fit run that the targets lack, because their
    ``(Step, Sequence, StepTime)`` grid or wavelengths differ, become
    missing values, which the fit run's imputation handles like any other
    missing value. Features the fit run does not have are dropped.

    Parameters
    ----------
    flattened : pl.DataFrame
        Flattened transform targets with a ``source`` column.
    features : Sequence[str]
        Feature columns of the fit run, in ``features.parquet`` order.

    Returns
    -------
    pl.DataFrame
        ``source`` followed by ``features`` in that order.

    Raises
    ------
    ValueError
        If the targets have none of the fit run's features.
    """
    present = set(flattened_feature_columns(flattened.columns))
    missing = [feature for feature in features if feature not in present]
    if len(missing) == len(features):
        raise ValueError("the transform targets have none of the fit run's features")
    if missing:
        logger.warning(
            f"{len(missing)} of {len(features)} features of the fit run are missing "
            "from the transform targets and are imputed"
        )
    return flattened.select(
        SOURCE_COLUMN,
        *(
            pl.col(feature) if feature in present else pl.lit(None, pl.Float64).alias(feature)
            for feature in features
        ),
    )


def transform_target_intensity(
    aligned: pl.DataFrame, features: Sequence[str], preprocess: Mapping[str, object]
) -> pl.DataFrame:
    """Apply the fit run's intensity transform to the aligned fit features.

    The transform is element-wise, so applying it after selecting the fit
    run's features gives the values of applying it before, as the fit job
    does, while only those values are checked against its domain.

    Parameters
    ----------
    aligned : pl.DataFrame
        ``source`` and ``features`` from ``align_features``, preprocessed
        without the intensity transform.
    features : Sequence[str]
        Feature columns of the fit run.
    preprocess : Mapping[str, object]
        The fit run's ``preprocess`` settings, with ``intensity_transform``
        and ``intensity_transform_scale`` (defaults ``"none"`` and 1.0).

    Returns
    -------
    pl.DataFrame
        ``source`` and the transformed ``float64`` features; missing values
        stay missing.

    Raises
    ------
    ValueError
        If ``log1p`` meets a value of the fit features outside its domain.
    """
    name = cast(IntensityTransform, preprocess.get("intensity_transform", "none"))
    if name == "none":
        return aligned
    values = transform_intensity_values(
        aligned.select(features).to_numpy(),
        name,
        float(cast(float, preprocess.get("intensity_transform_scale", 1.0))),
    )
    return pl.from_numpy(values, schema=list(features), orient="row").insert_column(
        0, aligned[SOURCE_COLUMN]
    )


def run_transform(config: dict[str, object], run_dir: Path) -> None:
    """Preprocess, flatten, transform, score, and save the target files.

    The targets are preprocessed with the fit run's settings, except that
    neither downsampling nor the missing ratio among the targets prunes a
    feature column (``TARGET_PREPROCESS_OVERRIDES``): the targets' grid may
    differ from the fit run's, so downsampling it would drop grid points the
    fit run kept. ``align_features`` then keeps the fit run's features, and
    ``transform_target_intensity`` applies the intensity transform to them.

    Parameters
    ----------
    config : dict[str, object]
        Configuration built by ``build_transform_config``.
    run_dir : Path
        Transform run directory receiving the artifacts and
        ``progress.json``.

    Raises
    ------
    ValueError
        If the fit run's settings are invalid for the targets, e.g.
        ``log1p`` meets a value outside its domain, the targets have none of
        the fit run's features, or the imputation drops every target.
    RunArtifactError
        If the fit run's model cannot be read.
    """
    files = cast(list[dict[str, object]], config["files"])
    dtype = np.dtype(str(config["artifact_dtype"]))

    def _stage(index: int) -> None:
        """Record the start of one stage."""
        logger.info(f"stage {STAGES[index]}")
        write_progress(run_dir, STAGES[index], index, len(STAGES))

    _stage(0)
    preprocess = {
        **cast(dict[str, object], config["preprocess"]),
        **TARGET_PREPROCESS_OVERRIDES,
    }
    flattened = preprocess_and_flatten(
        [Path(str(file["path"])) for file in files],
        **preprocess,  # type: ignore[arg-type]
        stem_uniqueness="error",
    ).collect()
    model = load_pca_model(Path(str(config[FIT_RUN_DIR_KEY])))
    aligned = transform_target_intensity(
        align_features(flattened, model.columns),
        model.columns,
        cast(dict[str, object], config["preprocess"]),
    )
    del flattened
    logger.info(f"flattened {aligned.height} files onto {len(model.columns)} features")

    _stage(1)
    scores = score_frame(aligned.lazy(), model, config)

    _stage(2)
    sample_frame(aligned[SOURCE_COLUMN].to_list(), config).write_parquet(
        run_dir / SAMPLES_FILE
    )
    np.save(
        run_dir / X_FILE,
        aligned.select(model.columns).to_numpy().astype(dtype, copy=False),
    )
    scores.write_parquet(run_dir / SCORES_FILE)
    write_progress(run_dir, STAGES[-1], len(STAGES), len(STAGES))
