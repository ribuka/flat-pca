"""Tests for the summary of a fit run's settings."""

from __future__ import annotations

import json

import pytest
from preprocess_settings import preprocess_settings

from flat_pca.webui.services.model_settings import NONE_TEXT, model_settings

STATISTICS = {"cumulative_explained_variance": 0.9, "alpha": 0.01}
_PCA = {
    "n_component": None,
    "impute_strategy": "drop",
    "impute_kmeans_n_clusters": None,
    "scaling_strategy": "none",
}


def _config(**overrides: object) -> dict[str, object]:
    """Return a fit configuration with some top-level entries replaced."""
    return {
        "files": [],
        "metadata_columns": [],
        "preprocess": preprocess_settings(),
        "pca": _PCA,
        "mahalanobis": STATISTICS,
        "spe": STATISTICS,
        "artifact_dtype": "float32",
        **overrides,
    }


def _run(config: object, **columns: object) -> dict[str, object]:
    """Return a fit run's ``runs`` row saving ``config``."""
    return {"run_id": "fit-1", "config_json": json.dumps(config), **columns}


def _rows(run: dict[str, object]) -> dict[str, dict[str, str]]:
    """Return the summary's rows as ``{section: {label: value}}``."""
    summary = model_settings(run)
    assert summary.error is None
    return {section.title: dict(section.rows) for section in summary.sections}


def test_unset_settings_are_shown_as_none() -> None:
    """Every row is shown; disabled ranges, windows, and edge trim are "なし"."""
    rows = _rows(_run(_config()))

    assert list(rows) == [
        "Targets",
        "Target Steps",
        "Preprocessing",
        "PCA",
        "T² / Q",
        "Artifacts",
    ]
    assert rows["Targets"] == {
        "Files": NONE_TEXT,
        "Features": NONE_TEXT,
        "Fitted components": NONE_TEXT,
    }
    assert rows["Target Steps"] == {"Target Steps": "1, 2"}
    assert rows["Preprocessing"] == {
        "Wavelength range": NONE_TEXT,
        "Edge trim (StepTime)": NONE_TEXT,
        "Time smoothing (half width, Time)": NONE_TEXT,
        "Wavelength smoothing (half width, nm)": NONE_TEXT,
        "Time normalization range (Time)": NONE_TEXT,
        "Wavelength normalization range (nm)": NONE_TEXT,
        "Intensity transform": "none",
        "Transform scale": "1",
        "Time downsampling stride": "1",
        "Wavelength downsampling stride": "1",
        "max_null_ratio": "0.1",
    }
    assert rows["PCA"] == {
        "Components": "auto",
        "Imputation": "drop",
        "kmeans clusters": NONE_TEXT,
        "Scaling": "none",
    }
    assert rows["T² / Q"] == {
        "T² cumulative explained variance": "0.9",
        "T² α": "0.01",
        "Q cumulative explained variance": "0.9",
        "Q α": "0.01",
    }
    assert rows["Artifacts"] == {"Artifact dtype": "float32"}
    assert model_settings(_run(_config())).headline == "auto components · scaling none"


def test_set_settings_are_shown_with_the_run_counts() -> None:
    """Enabled stages show their values; the counts come from the ``runs`` row."""
    config = _config(
        preprocess=preprocess_settings(
            target_steps=[2],
            edge_trim=[0.5, 0.25],
            wavelength_range=[400.0, 402.5],
            t_normalization_range=[0.0, 3.0],
            w_smoothing_window=1.5,
            intensity_transform="sqrt",
            intensity_transform_scale=2.0,
            t_downsampling_stride=2,
        ),
        pca={
            "n_component": 4,
            "impute_strategy": "kmeans",
            "impute_kmeans_n_clusters": 3,
            "scaling_strategy": "z-score",
        },
    )
    run = _run(config, n_files=12, n_features=3456, n_components=4)

    rows = _rows(run)

    assert rows["Targets"] == {
        "Files": "12",
        "Features": "3456",
        "Fitted components": "4",
    }
    assert rows["Target Steps"] == {"Target Steps": "2"}
    preprocessing = rows["Preprocessing"]
    assert preprocessing["Wavelength range"] == "400 – 402.5"
    assert preprocessing["Edge trim (StepTime)"] == "0.5 – 0.25"
    assert preprocessing["Time normalization range (Time)"] == "0 – 3"
    assert preprocessing["Wavelength normalization range (nm)"] == NONE_TEXT
    assert preprocessing["Time smoothing (half width, Time)"] == NONE_TEXT
    assert preprocessing["Wavelength smoothing (half width, nm)"] == "1.5"
    assert preprocessing["Intensity transform"] == "sqrt"
    assert preprocessing["Transform scale"] == "2"
    assert preprocessing["Time downsampling stride"] == "2"
    assert rows["PCA"] == {
        "Components": "4",
        "Imputation": "kmeans",
        "kmeans clusters": "3",
        "Scaling": "z-score",
    }
    assert model_settings(run).headline == "4 components · scaling z-score"


def test_kmeans_without_a_cluster_count_uses_the_default() -> None:
    """kmeans imputation without a cluster count shows "default"."""
    pca = {
        "n_component": None,
        "impute_strategy": "kmeans",
        "impute_kmeans_n_clusters": None,
        "scaling_strategy": "none",
    }

    assert _rows(_run(_config(pca=pca)))["PCA"]["kmeans clusters"] == "default"


def test_statistics_may_select_a_component_count() -> None:
    """T² and Q may select a component count, which the form cannot enter."""
    statistic = {"cumulative_explained_variance": 2, "alpha": 0.05}

    rows = _rows(_run(_config(mahalanobis=statistic)))

    assert rows["T² / Q"]["T² cumulative explained variance"] == "2"
    assert rows["T² / Q"]["T² α"] == "0.05"


@pytest.mark.parametrize(
    "config",
    [
        {key: value for key, value in _config().items() if key != "spe"},
        _config(preprocess=preprocess_settings(wavelength_range=400.0)),
        _config(pca=None),
        [],
        _config(preprocess=preprocess_settings(target_steps="12")),
        _config(pca={**_PCA, "scaling_strategy": []}),
        _config(pca={**_PCA, "impute_strategy": {}}),
        _config(pca={**_PCA, "n_component": "3"}),
        _config(preprocess=preprocess_settings(max_null_ratio=2.0)),
        _config(artifact_dtype="int8"),
        _config(spe={"cumulative_explained_variance": "0.9", "alpha": 0.01}),
        _config(spe={"cumulative_explained_variance": 0.9, "alpha": 1.5}),
        _config(pca={**_PCA, "n_component": True}),
        _config(preprocess=preprocess_settings(t_downsampling_stride=True)),
        _config(preprocess=preprocess_settings(max_null_ratio=False)),
        _config(preprocess=preprocess_settings(target_steps=[True, 2])),
        _config(spe={"cumulative_explained_variance": -1, "alpha": 0.01}),
        _config(spe={"cumulative_explained_variance": 1.5, "alpha": 0.01}),
        _config(mahalanobis={"cumulative_explained_variance": 0, "alpha": 0.01}),
        _config(mahalanobis={"cumulative_explained_variance": True, "alpha": 0.01}),
    ],
    ids=[
        "missing-key",
        "wrong-range",
        "wrong-section",
        "not-an-object",
        "steps-as-text",
        "scaling-as-list",
        "imputation-as-object",
        "components-as-text",
        "invalid-ratio",
        "invalid-dtype",
        "selector-as-text",
        "invalid-alpha",
        "components-as-bool",
        "stride-as-bool",
        "ratio-as-bool",
        "step-as-bool",
        "negative-count",
        "ratio-over-one",
        "zero-selector",
        "selector-as-bool",
    ],
)
def test_unreadable_settings_report_an_error(config: object) -> None:
    """A missing key or an invalid value becomes the summary's error."""
    summary = model_settings(_run(config))

    assert summary.run_id == "fit-1"
    assert summary.error is not None
    assert summary.error.startswith("Cannot read the settings of fit-1: ")
    assert summary.sections == ()


def test_unparsable_config_reports_an_error() -> None:
    """Saved configuration that is not JSON becomes the summary's error."""
    summary = model_settings({"run_id": "fit-1", "config_json": "{"})

    assert summary.error is not None
