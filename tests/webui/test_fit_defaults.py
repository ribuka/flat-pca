"""Tests for loading and validating ``fit_defaults.toml``."""

from __future__ import annotations

from pathlib import Path

import pytest

from flat_pca.webui.fit_defaults import (
    BUILTIN_FIT_DEFAULTS,
    FitDefaults,
    load_fit_defaults,
)
from flat_pca.webui.services.catalog_query import SelectionRanges
from flat_pca.webui.services.fit_form import default_form_values, parse_fit_form
from flat_pca.webui.settings import SettingsError

REPOSITORY_CONFIG = Path(__file__).resolve().parents[2] / "config"
RANGES = SelectionRanges(
    steps=[1, 2],
    wavelength_min=400.0,
    wavelength_max=402.5,
    time_min=0.0,
    time_max=3.0,
    step_time_max=1.0,
)


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    """Return a settings file path whose directory holds the defaults files.

    The settings file itself is not needed by ``load_fit_defaults``.
    """
    return tmp_path / "settings.toml"


def _write(settings_path: Path, text: str, name: str = "fit_defaults.toml") -> None:
    """Write a defaults file next to the settings file."""
    (settings_path.parent / name).write_text(text, encoding="utf-8")


def test_missing_file_uses_builtin_defaults(settings_path: Path) -> None:
    """Without a defaults file, the form keeps its built-in initial values."""
    defaults = load_fit_defaults(settings_path)

    assert defaults == BUILTIN_FIT_DEFAULTS
    assert default_form_values(RANGES, defaults.form_values()) == default_form_values(RANGES)


def test_repository_example_keeps_builtin_defaults() -> None:
    """The commented example in ``config/`` loads and changes nothing."""
    assert load_fit_defaults(REPOSITORY_CONFIG / "settings.toml") == BUILTIN_FIT_DEFAULTS


def test_partial_file_replaces_only_the_given_values(settings_path: Path) -> None:
    """Given values replace the initial form values; the rest stay built in."""
    _write(
        settings_path,
        "[preprocess]\n"
        "edge_trim_start = 0.5\n"
        "intensity_transform = 'log1p'\n"
        "intensity_transform_scale = 2\n"
        "w_downsampling_stride = 3\n"
        "[preprocess.wavelength_range]\n"
        "enabled = true\n"
        "lower = 401\n"
        "[pca]\n"
        "n_component = 4\n"
        "impute_strategy = 'kmeans'\n"
        "scaling_strategy = 'z-score'\n"
        "[spe]\n"
        "alpha = 0.05\n",
    )

    values = default_form_values(RANGES, load_fit_defaults(settings_path).form_values())

    assert values == {
        **default_form_values(RANGES),
        "edge_trim_start": "0.5",
        "intensity_transform": "log1p",
        "intensity_transform_scale": "2",
        "w_downsampling_stride": "3",
        "wavelength_range_enabled": True,
        "wavelength_range_lower": "401",
        "n_component": "4",
        "impute_strategy": "kmeans",
        "scaling_strategy": "z-score",
        "spe_alpha": "0.05",
    }
    form = parse_fit_form(values, n_files=4)
    assert form.is_valid
    assert form.preprocess["edge_trim"] == [0.5, 0.0]
    assert form.preprocess["wavelength_range"] == [401.0, 402.5]
    assert form.pca["n_component"] == 4
    assert form.spe == {"cumulative_explained_variance": 0.9, "alpha": 0.05}


def test_local_file_replaces_the_shared_file(settings_path: Path) -> None:
    """``fit_defaults.local.toml`` is read instead of, not merged with, the shared file."""
    _write(settings_path, "[pca]\nscaling_strategy = 'pareto'\nn_component = 2\n")
    _write(settings_path, "[pca]\nn_component = 3\n", "fit_defaults.local.toml")

    defaults = load_fit_defaults(settings_path)

    assert defaults.pca.n_component == 3
    assert defaults.pca.scaling_strategy is None


def test_local_file_is_read_without_the_shared_file(settings_path: Path) -> None:
    """The local file alone is enough."""
    _write(settings_path, "[mahalanobis]\nalpha = 0.02\n", "fit_defaults.local.toml")

    assert load_fit_defaults(settings_path).mahalanobis.alpha == 0.02


def test_file_is_looked_up_next_to_the_settings_file(tmp_path: Path) -> None:
    """The defaults follow the directory of ``--settings``, not the working directory."""
    other = tmp_path / "other"
    other.mkdir()
    _write(tmp_path / "settings.toml", "[pca]\nn_component = 2\n")

    assert load_fit_defaults(other / "settings.toml") == BUILTIN_FIT_DEFAULTS
    assert load_fit_defaults(tmp_path / "settings.toml").pca.n_component == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[pca]\nunknown = 1\n", "pca.unknown"),
        ("[unknown]\n", "unknown"),
        ("[preprocess]\ntarget_steps = [1]\n", "target_steps"),
        ("[preprocess.wavelength_range]\nmin = 1\n", "min"),
        ("[preprocess]\nt_downsampling_stride = 1.5\n", "t_downsampling_stride"),
        ("[preprocess]\nmax_null_ratio = '0.1'\n", "max_null_ratio"),
        ("[preprocess.wavelength_range]\nenabled = 1\n", "enabled"),
        ("[preprocess.wavelength_range]\nenabled = true\nlower = nan\n", "lower"),
        ("[preprocess.t_normalization_range]\nupper = inf\n", "upper"),
        ("[preprocess.w_normalization_range]\nlower = -inf\nupper = 1\n", "lower"),
        ("[pca]\nn_component = 0\n", "n_component"),
        ("[pca]\nimpute_kmeans_n_clusters = 0\n", "impute_kmeans_n_clusters"),
    ],
)
def test_unknown_keys_and_wrong_types_are_rejected(
    settings_path: Path, body: str, message: str
) -> None:
    """Unknown keys and values of the wrong type are errors."""
    _write(settings_path, body)

    with pytest.raises(SettingsError, match=message):
        load_fit_defaults(settings_path)


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ("[preprocess]\nintensity_transform = 'cube'\n", "preprocess.intensity_transform"),
        (
            "[preprocess]\nintensity_transform = 'log1p'\nintensity_transform_scale = 0\n",
            "preprocess.intensity_transform_scale",
        ),
        ("[preprocess]\nedge_trim_end = inf\n", "preprocess.edge_trim"),
        ("[preprocess]\nt_smoothing_window = -1\n", "preprocess.t_smoothing_window"),
        ("[preprocess]\nw_downsampling_stride = 0\n", "preprocess.w_downsampling_stride"),
        ("[preprocess]\nmax_null_ratio = 2.0\n", "preprocess.max_null_ratio"),
        (
            "[preprocess.t_normalization_range]\nlower = 5\nupper = 1\n",
            "preprocess.t_normalization_range",
        ),
        ("[pca]\nimpute_strategy = 'mean'\n", "pca.impute_strategy"),
        ("[pca]\nscaling_strategy = 'unit'\n", "pca.scaling_strategy"),
        ("[mahalanobis]\nalpha = 1.0\n", "mahalanobis.alpha"),
        ("[spe]\ncumulative_explained_variance = 0\n", "spe.cumulative_explained_variance"),
    ],
)
def test_values_rejected_by_the_form_are_errors(
    settings_path: Path, body: str, key: str
) -> None:
    """Values follow the rules of the form and name the offending key."""
    _write(settings_path, body)

    with pytest.raises(SettingsError, match=rf"(?m)^{key}: "):
        load_fit_defaults(settings_path)


def test_range_with_one_bound_is_checked_on_the_form(settings_path: Path) -> None:
    """A single bound is accepted; the catalog supplies the other on the page."""
    _write(settings_path, "[preprocess.wavelength_range]\nenabled = true\nupper = 300\n")

    defaults = load_fit_defaults(settings_path)

    form = parse_fit_form(default_form_values(RANGES, defaults.form_values()), n_files=2)
    assert set(form.errors) == {"wavelength_range"}


def test_choices_are_stripped_as_the_form_reads_them(settings_path: Path) -> None:
    """Choices with surrounding spaces become the exact option values."""
    _write(
        settings_path,
        "[preprocess]\nintensity_transform = ' sqrt '\n"
        "[pca]\nimpute_strategy = 'median '\nscaling_strategy = ' z-score'\n",
    )

    values = load_fit_defaults(settings_path).form_values()

    assert values["intensity_transform"] == "sqrt"
    assert values["impute_strategy"] == "median"
    assert values["scaling_strategy"] == "z-score"


def test_invalid_toml_is_an_error(settings_path: Path) -> None:
    """A file that is not TOML is reported with its path."""
    _write(settings_path, "[pca\n")

    with pytest.raises(SettingsError, match="invalid TOML"):
        load_fit_defaults(settings_path)


def test_form_values_hold_only_given_values() -> None:
    """Only the given values and the enabled state of the ranges are replaced."""
    defaults = FitDefaults.model_validate(
        {"preprocess": {"t_normalization_range": {"enabled": True}}, "mahalanobis": {"alpha": 0.1}}
    )

    assert defaults.form_values() == {
        "wavelength_range_enabled": False,
        "t_normalization_range_enabled": True,
        "w_normalization_range_enabled": False,
        "mahalanobis_alpha": "0.1",
    }
