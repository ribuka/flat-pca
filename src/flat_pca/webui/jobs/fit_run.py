"""Fit job: preprocess, flatten, and fit PCA to the selected files.

The job runs in a child process and never touches the database. It writes
the run artifacts (see ``docs/spec/webui.md``) to the run directory; the
app process validates them with ``services.fit_artifacts.register_fit_result``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from loguru import logger

from flat_pca.feature_engineering.flatten_pca import preprocess_and_flatten
from flat_pca.feature_engineering.pca import (
    MahalanobisConfig,
    PcaModel,
    SpeConfig,
    fit_pca,
    transform_pca,
    truncate_pca_model,
)
from flat_pca.feature_engineering.pca.mahalanobis import resolve_mahalanobis_components
from flat_pca.spectral.schema import (
    SOURCE_COLUMN,
    flattened_feature_columns,
    parse_feature_coordinate,
)

from ..settings import MetadataColumnType, Settings
from .progress import write_progress

FEATURES_FILE = "features.parquet"
SAMPLES_FILE = "samples.parquet"
X_FILE = "X.npy"
COMPONENTS_FILE = "components.npy"
PCA_STATE_FILE = "pca_state.npz"
SCORES_FILE = "scores.parquet"

# Without an explicit n_component, keep the components reaching this
# cumulative explained-variance ratio, fitting at most AUTO_COMPONENT_CAP.
AUTO_COMPONENT_CUMULATIVE = 0.99
AUTO_COMPONENT_CAP = 1000
STAGES = ("preprocess", "fit", "save")

_METADATA_DTYPES: dict[MetadataColumnType, pl.DataType] = {
    "category": pl.String(),
    "number": pl.Float64(),
    "datetime": pl.Datetime("us"),
}


def build_fit_config(
    settings: Settings,
    files: Sequence[Mapping[str, object]],
    preprocess: Mapping[str, object],
    pca: Mapping[str, object],
    mahalanobis: Mapping[str, object],
    spe: Mapping[str, object],
) -> dict[str, object]:
    """Build the JSON configuration of a fit job.

    Parameters
    ----------
    settings : Settings
        Application settings, giving the metadata columns and
        ``jobs.artifact_dtype``.
    files : Sequence[Mapping[str, object]]
        Target files with ``stem``, ``path``, and every metadata column.
    preprocess : Mapping[str, object]
        Validated ``preprocess_and_flatten`` keyword arguments.
    pca : Mapping[str, object]
        Validated ``n_component`` (``None`` for the automatic count),
        ``impute_strategy``, ``impute_kmeans_n_clusters``, and
        ``scaling_strategy``.
    mahalanobis : Mapping[str, object]
        Validated ``MahalanobisConfig`` keyword arguments.
    spe : Mapping[str, object]
        Validated ``SpeConfig`` keyword arguments.

    Returns
    -------
    dict[str, object]
        Configuration passed to ``run_fit`` and saved as ``config.json``.
        Datetime metadata values are stored as ISO 8601 text.
    """
    return {
        **file_entries(settings, files),
        "preprocess": dict(preprocess),
        "pca": dict(pca),
        "mahalanobis": dict(mahalanobis),
        "spe": dict(spe),
        "artifact_dtype": settings.jobs.artifact_dtype,
    }


def file_entries(
    settings: Settings, files: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Build the target-file entries of a job configuration.

    Parameters
    ----------
    settings : Settings
        Application settings, giving the metadata columns.
    files : Sequence[Mapping[str, object]]
        Target files with ``stem``, ``path``, and every metadata column.

    Returns
    -------
    dict[str, object]
        ``files`` (each file's ``stem``, ``path``, and ``metadata``, with
        datetime values as ISO 8601 text) and ``metadata_columns`` (each
        column's type), as read by ``sample_frame``.
    """
    columns = settings.metadata_columns
    return {
        "files": [
            {
                "stem": str(file["stem"]),
                "path": str(file["path"]),
                "metadata": {
                    name: _json_value(file.get(name)) for name in columns
                },
            }
            for file in files
        ],
        "metadata_columns": {name: column.type for name, column in columns.items()},
    }


def _json_value(value: object) -> object:
    """Convert a metadata value to a JSON-compatible value.

    Parameters
    ----------
    value : object
        Metadata value from the catalog.

    Returns
    -------
    object
        ISO 8601 text for values with ``isoformat``, otherwise ``value``.
    """
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else value


def fit_run_pca(
    flattened: pl.LazyFrame, features: list[str], settings: Mapping[str, object]
) -> PcaModel:
    """Fit the PCA pipeline of a fit run.

    Outlier handling is not applied, as in ``flatten_pca``. Without an
    explicit ``n_component``, at most ``AUTO_COMPONENT_CAP`` components are
    fitted and the leading ones reaching ``AUTO_COMPONENT_CUMULATIVE`` of
    the explained variance are kept (see ``truncate_pca_model``).

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened features with a ``source`` column.
    features : list[str]
        Feature columns of ``flattened``.
    settings : Mapping[str, object]
        ``pca`` entry of the job configuration.

    Returns
    -------
    PcaModel
        Fitted PCA pipeline.

    Raises
    ------
    ValueError
        If the settings are invalid for the data.
    """
    n_component = cast(int | None, settings["n_component"])
    if n_component is not None and n_component > len(features):
        raise ValueError(
            f"n_component ({n_component}) exceeds the feature count ({len(features)})"
        )
    model = fit_pca(
        flattened,
        features,
        n_component=n_component,
        max_n_component=AUTO_COMPONENT_CAP if n_component is None else None,
        impute_strategy=settings["impute_strategy"],  # type: ignore[arg-type]
        outlier_strategy=None,
        scaling_strategy=settings["scaling_strategy"],  # type: ignore[arg-type]
        impute_kmeans_n_clusters=cast(int | None, settings["impute_kmeans_n_clusters"]),
    )
    if n_component is None:
        model = truncate_pca_model(
            model, resolve_mahalanobis_components(model.pca, AUTO_COMPONENT_CUMULATIVE)
        )
    return model


def feature_frame(features: Sequence[str]) -> pl.DataFrame:
    """Decode flattened feature names into their coordinates.

    Parameters
    ----------
    features : Sequence[str]
        Flattened feature names, in ``X.npy`` column order.

    Returns
    -------
    pl.DataFrame
        ``feature``, ``wavelength``, ``Step``, ``Sequence``, and ``StepTime``,
        one row per feature in the given order.
    """
    coordinates = [parse_feature_coordinate(feature) for feature in features]
    return pl.DataFrame(
        {
            "feature": list(features),
            "wavelength": [coordinate[0] for coordinate in coordinates],
            "Step": [coordinate[1] for coordinate in coordinates],
            "Sequence": [coordinate[2] for coordinate in coordinates],
            "StepTime": [coordinate[3] for coordinate in coordinates],
        },
        schema={
            "feature": pl.String,
            "wavelength": pl.Float64,
            "Step": pl.Int64,
            "Sequence": pl.Int64,
            "StepTime": pl.Float64,
        },
    )


def sample_frame(sources: Sequence[str], config: Mapping[str, object]) -> pl.DataFrame:
    """Build the per-sample table with each file's metadata.

    Parameters
    ----------
    sources : Sequence[str]
        ``source`` values in ``X.npy`` row order.
    config : Mapping[str, object]
        Job configuration holding the entries of ``file_entries``.

    Returns
    -------
    pl.DataFrame
        ``source``, ``stem``, and the metadata columns typed by their
        configured type, one row per source in the given order.

    Raises
    ------
    ValueError
        If a source is not one of the configured files.
    """
    files = cast(list[dict[str, object]], config["files"])
    metadata_by_stem = {
        str(file["stem"]): cast(dict[str, object], file["metadata"]) for file in files
    }
    column_types = cast(dict[str, MetadataColumnType], config["metadata_columns"])
    stems = [Path(source).stem for source in sources]
    unknown = [stem for stem in stems if stem not in metadata_by_stem]
    if unknown:
        raise ValueError(f"flattened rows of unknown files: {unknown}")
    columns: list[pl.Series] = [
        pl.Series(SOURCE_COLUMN, list(sources), dtype=pl.String),
        pl.Series("stem", stems, dtype=pl.String),
    ]
    for name, column_type in column_types.items():
        values = [metadata_by_stem[stem].get(name) for stem in stems]
        if column_type == "datetime":
            series = pl.Series(name, values, dtype=pl.String).str.to_datetime(
                time_unit="us"
            )
        else:
            series = pl.Series(name, values, dtype=_METADATA_DTYPES[column_type])
        columns.append(series)
    return pl.DataFrame(columns)


def score_frame(
    flattened: pl.LazyFrame, model: PcaModel, config: Mapping[str, object]
) -> pl.DataFrame:
    """Score the flattened rows with T² and Q statistics.

    Parameters
    ----------
    flattened : pl.LazyFrame
        Flattened features with a ``source`` column.
    model : PcaModel
        Fitted PCA pipeline.
    config : Mapping[str, object]
        Job configuration with ``mahalanobis`` and ``spe`` settings.

    Returns
    -------
    pl.DataFrame
        ``source``, the score columns, and the Mahalanobis (T²) and SPE (Q)
        columns. Rows dropped by ``impute_strategy="drop"`` are absent.
    """
    mahalanobis = MahalanobisConfig(**cast(dict[str, object], config["mahalanobis"]))  # type: ignore[arg-type]
    spe = SpeConfig(**cast(dict[str, object], config["spe"]))  # type: ignore[arg-type]
    return (
        transform_pca(flattened, model, mahalanobis=mahalanobis, spe=spe)
        .select(
            SOURCE_COLUMN,
            *model.pca_column_names,
            *mahalanobis.column_names,
            *spe.column_names,
        )
        .collect()
    )


def run_fit(config: dict[str, object], run_dir: Path) -> None:
    """Preprocess, flatten, fit PCA, and save the run artifacts.

    The fitted files are not scored: their scores, T², and Q belong to a
    transform run (``jobs.transform_run.run_transform``).

    Parameters
    ----------
    config : dict[str, object]
        Configuration built by ``build_fit_config``.
    run_dir : Path
        Run directory receiving the artifacts and ``progress.json``.

    Raises
    ------
    ValueError
        If the settings are invalid for the data, e.g. ``log1p`` meets a
        value outside its domain or no feature column remains.
    """
    files = cast(list[dict[str, object]], config["files"])
    dtype = np.dtype(str(config["artifact_dtype"]))

    def _stage(index: int) -> None:
        """Record the start of one stage."""
        logger.info(f"stage {STAGES[index]}")
        write_progress(run_dir, STAGES[index], index, len(STAGES))

    _stage(0)
    flattened = preprocess_and_flatten(
        [Path(str(file["path"])) for file in files],
        **cast(dict[str, object], config["preprocess"]),  # type: ignore[arg-type]
        stem_uniqueness="error",
    ).collect()
    features = flattened_feature_columns(flattened.columns)
    if not features:
        raise ValueError("no feature column remains after preprocessing")
    logger.info(f"flattened {flattened.height} files into {len(features)} features")

    _stage(1)
    model = fit_run_pca(flattened.lazy(), features, cast(dict[str, object], config["pca"]))
    logger.info(f"fitted {model.n_component} components")

    _stage(2)
    feature_frame(features).write_parquet(run_dir / FEATURES_FILE)
    sample_frame(flattened[SOURCE_COLUMN].to_list(), config).write_parquet(
        run_dir / SAMPLES_FILE
    )
    np.save(
        run_dir / X_FILE,
        flattened.select(features).to_numpy().astype(dtype, copy=False),
    )
    del flattened
    np.save(run_dir / COMPONENTS_FILE, model.pca.components_.astype(dtype))
    np.savez(run_dir / PCA_STATE_FILE, **model.to_pca_state())
    write_progress(run_dir, STAGES[-1], len(STAGES), len(STAGES))
