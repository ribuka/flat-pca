"""Compare T², Q, and reconstructions of fit artifacts saved in float32 and float64.

Run on a local fit run of real data to check how saving ``X.npy`` and
``components.npy`` in ``float32`` (``jobs.artifact_dtype``) affects the
values computed from them::

    uv run -m flat_pca.webui.compare_artifact_dtypes --run-dir <workspace>/runs/<run_id>

The run's ``config.json`` is fitted again once, and the result is saved in
``--out-dir`` twice: with ``float64`` matrices and with ``float32`` ones.
Each copy is restored from its artifacts and recomputed in ``float64``, and
the differences between the two are printed. Fitting once keeps the
randomized PCA solver from adding differences of its own.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from loguru import logger

from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig, transform_pca
from flat_pca.spectral.schema import SOURCE_COLUMN

from .jobs.executor import CONFIG_FILE
from .jobs.fit_run import COMPONENTS_FILE, X_FILE, run_fit
from .services.fit_artifacts import FitArtifacts, load_fit_artifacts


def recompute(
    artifacts: FitArtifacts, config: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Recompute scores, T², Q, and reconstructions from saved artifacts.

    Parameters
    ----------
    artifacts : FitArtifacts
        Loaded artifacts of one run.
    config : Mapping[str, object]
        The run's configuration, giving the T² and Q settings.

    Returns
    -------
    dict[str, np.ndarray]
        ``scores``, ``t2``, ``q``, and ``reconstruction`` (every component),
        one row per file kept by the imputation stage, in ``float64``.
    """
    features = artifacts.features["feature"].to_list()
    frame = pl.DataFrame(
        np.asarray(artifacts.x, dtype=np.float64), schema=features
    ).insert_column(0, artifacts.samples[SOURCE_COLUMN])
    mahalanobis = MahalanobisConfig(**cast(dict[str, object], config["mahalanobis"]))  # type: ignore[arg-type]
    spe = SpeConfig(**cast(dict[str, object], config["spe"]))  # type: ignore[arg-type]
    model = artifacts.model
    scored = transform_pca(
        frame.lazy(), model, mahalanobis=mahalanobis, spe=spe
    ).collect()
    scores = scored.select(model.pca_column_names)
    return {
        "scores": scores.to_numpy(),
        "t2": scored[mahalanobis.distance_column].to_numpy(),
        "q": scored[spe.spe_column].to_numpy(),
        "reconstruction": model.reconstruct(scores).select(features).to_numpy(),
    }


def compare_values(
    single: Mapping[str, np.ndarray], double: Mapping[str, np.ndarray]
) -> pl.DataFrame:
    """Summarize the differences between float32- and float64-based values.

    Parameters
    ----------
    single : Mapping[str, np.ndarray]
        Values recomputed from ``float32`` artifacts.
    double : Mapping[str, np.ndarray]
        Values recomputed from ``float64`` artifacts.

    Returns
    -------
    pl.DataFrame
        One row per quantity with ``max_abs_diff``, ``max_abs_value`` (of
        the float64 values), and ``relative_diff`` (their ratio).
        Score signs are aligned per component first, since a component's
        sign is arbitrary.
    """
    rows = []
    for name, reference in double.items():
        values = single[name]
        if name == "scores":
            signs = np.sign(np.sum(values * reference, axis=0))
            values = values * np.where(signs == 0, 1.0, signs)
        max_abs_diff = float(np.max(np.abs(values - reference)))
        max_abs_value = float(np.max(np.abs(reference)))
        rows.append(
            {
                "quantity": name,
                "max_abs_diff": max_abs_diff,
                "max_abs_value": max_abs_value,
                "relative_diff": max_abs_diff / max_abs_value if max_abs_value else 0.0,
            }
        )
    return pl.DataFrame(rows)


def copy_as_float32(source_dir: Path, target_dir: Path) -> None:
    """Copy float64 artifacts, saving ``X.npy`` and ``components.npy`` in float32.

    Both runs then hold the same preprocessing and fit result, so comparing
    them measures only the precision lost by saving in ``float32``.

    Parameters
    ----------
    source_dir : Path
        Run directory written by ``run_fit`` with ``artifact_dtype="float64"``.
    target_dir : Path
        New directory receiving the copy.
    """
    shutil.copytree(source_dir, target_dir)
    for name in (X_FILE, COMPONENTS_FILE):
        values = np.load(source_dir / name)
        np.save(target_dir / name, values.astype(np.float32))


def compare_run(run_dir: Path, out_dir: Path) -> pl.DataFrame:
    """Refit a run once and compare its artifacts saved in float32 and float64.

    The run's configuration is fitted once with ``artifact_dtype="float64"``
    into ``{out_dir}/float64``, and ``{out_dir}/float32`` receives the same
    result with its matrices converted to ``float32``. Refitting per dtype
    would add differences from the randomized PCA solver.

    Parameters
    ----------
    run_dir : Path
        Fit run directory holding ``config.json``.
    out_dir : Path
        Directory receiving the two copies of the refitted run.

    Returns
    -------
    pl.DataFrame
        Differences summarized by ``compare_values``.
    """
    config = {
        **json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8")),
        "artifact_dtype": "float64",
    }
    double_dir = out_dir / "float64"
    single_dir = out_dir / "float32"
    double_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"fitting {run_dir} into {double_dir}")
    run_fit(config, double_dir)
    copy_as_float32(double_dir, single_dir)
    return compare_values(
        recompute(load_fit_artifacts(single_dir), config),
        recompute(load_fit_artifacts(double_dir), config),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, compare the dtypes, and print the differences.

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="fit run directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for the refitted run (default: <run-dir>/dtype-comparison)",
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.run_dir / "dtype-comparison"
    with pl.Config(tbl_rows=-1, fmt_float="mixed"):
        print(compare_run(args.run_dir, out_dir))


if __name__ == "__main__":
    main()
