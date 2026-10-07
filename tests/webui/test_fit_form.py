"""Tests for the defaults and validation of the preprocessing and PCA form."""

from __future__ import annotations

import pytest

from flat_pca.webui.jobs.fit_run import build_fit_config
from flat_pca.webui.services.catalog_query import SelectionRanges
from flat_pca.webui.services.fit_form import (
    default_form_values,
    form_values_from_config,
    parse_fit_form,
)
from flat_pca.webui.settings import Settings

RANGES = SelectionRanges(
    steps=[1, 2, 5],
    wavelength_min=400.0,
    wavelength_max=402.5,
    time_min=0.0,
    time_max=3.0,
    step_time_max=1.0,
)


def _values(**overrides: object) -> dict[str, object]:
    """Return the default form values with some fields replaced."""
    return {**default_form_values(RANGES), **overrides}


def test_default_values_parse_to_library_defaults() -> None:
    """The initial form selects every step and no optional stage."""
    form = parse_fit_form(default_form_values(RANGES), n_files=3)

    assert form.is_valid
    assert form.preprocess == {
        "target_steps": [1, 2, 5],
        "edge_trim": None,
        "wavelength_range": None,
        "t_normalization_range": None,
        "w_normalization_range": None,
        "t_smoothing_window": None,
        "w_smoothing_window": None,
        "intensity_transform": "none",
        "intensity_transform_scale": 1.0,
        "t_downsampling_stride": 1,
        "w_downsampling_stride": 1,
        "max_null_ratio": 0.1,
    }
    assert form.pca == {
        "n_component": None,
        "impute_strategy": "drop",
        "impute_kmeans_n_clusters": None,
        "scaling_strategy": "none",
    }
    assert form.mahalanobis == {"cumulative_explained_variance": 0.9, "alpha": 0.01}
    assert form.spe == {"cumulative_explained_variance": 0.9, "alpha": 0.01}


def test_default_ranges_come_from_the_catalog() -> None:
    """Disabled ranges start from the catalog's wavelength and Time ranges."""
    values = default_form_values(RANGES)

    assert values["wavelength_range_enabled"] is False
    assert (values["wavelength_range_lower"], values["wavelength_range_upper"]) == (
        "400",
        "402.5",
    )
    assert (values["t_normalization_range_lower"], values["t_normalization_range_upper"]) == (
        "0",
        "3",
    )


def test_enabled_stages_are_parsed() -> None:
    """Enabled ranges, trims, windows, and transforms become job settings."""
    form = parse_fit_form(
        _values(
            target_steps=["2", "1"],
            edge_trim_start="0.5",
            edge_trim_end="",
            wavelength_range_enabled="1",
            wavelength_range_lower="400",
            wavelength_range_upper="401",
            t_smoothing_window="0.25",
            intensity_transform="asinh",
            intensity_transform_scale="2",
            n_component="2",
            impute_strategy="kmeans",
            impute_kmeans_n_clusters="3",
            scaling_strategy="pareto",
            spe_cumulative_explained_variance="1",
        ),
        n_files=3,
    )

    assert form.is_valid, form.errors
    assert form.preprocess["target_steps"] == [1, 2]
    assert form.preprocess["edge_trim"] == [0.5, 0.0]
    assert form.preprocess["wavelength_range"] == [400.0, 401.0]
    assert form.preprocess["t_smoothing_window"] == 0.25
    assert form.preprocess["intensity_transform"] == "asinh"
    assert form.pca == {
        "n_component": 2,
        "impute_strategy": "kmeans",
        "impute_kmeans_n_clusters": 3,
        "scaling_strategy": "pareto",
    }
    assert form.spe["cumulative_explained_variance"] == 1.0


@pytest.mark.parametrize(
    ("overrides", "field", "message"),
    [
        ({"target_steps": []}, "target_steps", "Step を 1 つ以上"),
        ({"target_steps": ["x"]}, "target_steps", "整数"),
        ({"edge_trim_start": "a"}, "edge_trim", "数値"),
        (
            {
                "wavelength_range_enabled": "1",
                "wavelength_range_lower": "402",
                "wavelength_range_upper": "401",
            },
            "wavelength_range",
            "wavelength_range",
        ),
        (
            {"t_normalization_range_enabled": "1", "t_normalization_range_upper": ""},
            "t_normalization_range",
            "値を入力",
        ),
        ({"t_smoothing_window": "-1"}, "t_smoothing_window", "t_smoothing_window"),
        ({"intensity_transform": "exp"}, "intensity_transform", "いずれか"),
        (
            {"intensity_transform": "log1p", "intensity_transform_scale": "0"},
            "intensity_transform_scale",
            "intensity_transform_scale",
        ),
        ({"t_downsampling_stride": "1.5"}, "t_downsampling_stride", "整数"),
        ({"w_downsampling_stride": "0"}, "w_downsampling_stride", "w_downsampling_stride"),
        ({"max_null_ratio": "2"}, "max_null_ratio", "between 0.0 and 1.0"),
        ({"n_component": "4"}, "n_component", "選択ファイル数（3）以下"),
        ({"n_component": "0"}, "n_component", "1 以上"),
        ({"impute_strategy": "mean"}, "impute_strategy", "drop, median, kmeans"),
        (
            {"impute_strategy": "kmeans", "impute_kmeans_n_clusters": "0"},
            "impute_kmeans_n_clusters",
            "1 以上",
        ),
        ({"scaling_strategy": "log"}, "scaling_strategy", "pareto"),
        ({"mahalanobis_alpha": "1"}, "mahalanobis_alpha", "alpha"),
        (
            {"spe_cumulative_explained_variance": "1.5"},
            "spe_cumulative_explained_variance",
            "cumulative_explained_variance",
        ),
        ({"spe_alpha": ""}, "spe_alpha", "値を入力"),
    ],
)
def test_invalid_fields_get_their_own_error(
    overrides: dict[str, object], field: str, message: str
) -> None:
    """Each invalid field is reported under its own key."""
    form = parse_fit_form(_values(**overrides), n_files=3)

    assert list(form.errors) == [field]
    assert message in form.errors[field]


def test_disabled_ranges_and_unused_clusters_are_not_validated() -> None:
    """Values of disabled stages are ignored."""
    form = parse_fit_form(
        _values(
            wavelength_range_lower="abc",
            impute_strategy="median",
            impute_kmeans_n_clusters="abc",
        ),
        n_files=3,
    )

    assert form.is_valid
    assert form.preprocess["wavelength_range"] is None
    assert form.pca["impute_kmeans_n_clusters"] is None


def test_missing_selection_is_an_error() -> None:
    """Without selected files the form reports ``files``."""
    form = parse_fit_form(default_form_values(RANGES), n_files=0)

    assert "データ選択画面" in form.errors["files"]


def test_values_from_a_run_config_reproduce_its_settings(settings: Settings) -> None:
    """Reopening a run fills the form with the settings it ran with."""
    submitted = parse_fit_form(
        _values(
            target_steps=["2"],
            edge_trim_start="0.5",
            edge_trim_end="0.25",
            w_normalization_range_enabled="1",
            w_normalization_range_lower="400.5",
            w_normalization_range_upper="402",
            w_smoothing_window="1.5",
            intensity_transform="log1p",
            intensity_transform_scale="0.5",
            t_downsampling_stride="2",
            max_null_ratio="0.2",
            n_component="2",
            impute_strategy="kmeans",
            scaling_strategy="robust",
            mahalanobis_alpha="0.05",
        ),
        n_files=3,
    )
    assert submitted.is_valid
    config = build_fit_config(
        settings,
        [],
        submitted.preprocess,
        submitted.pca,
        submitted.mahalanobis,
        submitted.spe,
    )

    reopened = parse_fit_form(form_values_from_config(config, RANGES), n_files=3)

    assert reopened.is_valid
    assert reopened.preprocess == submitted.preprocess
    assert reopened.pca == submitted.pca
    assert reopened.mahalanobis == submitted.mahalanobis
    assert reopened.spe == submitted.spe
    assert reopened.values["wavelength_range_enabled"] is False
    assert reopened.values["wavelength_range_lower"] == "400"
