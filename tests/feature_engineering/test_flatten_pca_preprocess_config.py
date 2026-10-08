"""Tests for the validated Flatten-PCA preprocessing configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from flat_pca.feature_engineering import flatten_pca, preprocess_and_flatten
from flat_pca.feature_engineering.flatten_pca.preprocess_config import PreprocessConfig

INVALID_ARGUMENTS = [
    ("edge_trim", (0.0,)),
    ("edge_trim", (0.0, float("nan"))),
    ("wavelength_range", (2.0, 1.0)),
    ("t_smoothing_window", 0.0),
    ("w_smoothing_window", float("inf")),
    ("t_normalization_range", (1.0, 0.0)),
    ("w_normalization_range", "0,1"),
    ("intensity_transform", "log"),
    ("intensity_transform", None),
    ("intensity_transform_scale", 0.0),
    ("intensity_transform_scale", float("nan")),
    ("intensity_transform_scale", True),
    ("t_downsampling_stride", 0),
    ("w_downsampling_stride", True),
    ("max_null_ratio", 1.5),
]


def test_preprocess_config_defaults_match_public_api_defaults() -> None:
    """Default to the same values as the public keyword arguments."""
    config = PreprocessConfig()

    assert config.target_steps is None
    assert config.edge_trim is None
    assert config.wavelength_range is None
    assert config.t_smoothing_window is None
    assert config.w_smoothing_window is None
    assert config.t_normalization_range is None
    assert config.w_normalization_range is None
    assert config.intensity_transform == "none"
    assert config.intensity_transform_scale == 1.0
    assert config.t_downsampling_stride == 1
    assert config.w_downsampling_stride == 1
    assert config.max_null_ratio == 0.1
    assert config.stem_uniqueness == "skip"
    assert config.validate_metadata_uniqueness is False
    assert config.validate_metadata_alignment is False


def test_preprocess_config_stores_coerced_values() -> None:
    """Store validated values in the forms the preprocessing stages consume."""
    config = PreprocessConfig(
        edge_trim=[1, 2],
        wavelength_range=(400, 700),
        t_smoothing_window=2,
        w_smoothing_window=3,
        t_normalization_range=[0, 5],
        w_normalization_range=(500, 600),
        intensity_transform="asinh",
        intensity_transform_scale=2,
        t_downsampling_stride=np.int64(2),
        w_downsampling_stride=3,
    )

    assert config.edge_trim == (1.0, 2.0)
    assert config.wavelength_range == (400.0, 700.0)
    assert config.t_smoothing_window == 2.0
    assert isinstance(config.t_smoothing_window, float)
    assert config.w_smoothing_window == 3.0
    assert config.t_normalization_range == (0.0, 5.0)
    assert config.w_normalization_range == (500.0, 600.0)
    assert config.intensity_transform == "asinh"
    assert config.intensity_transform_scale == 2.0
    assert isinstance(config.intensity_transform_scale, float)
    assert config.t_downsampling_stride == 2
    assert type(config.t_downsampling_stride) is int
    assert config.w_downsampling_stride == 3


def test_preprocess_config_is_frozen() -> None:
    """Reject attribute assignment after construction."""
    config = PreprocessConfig()

    with pytest.raises(FrozenInstanceError):
        config.t_smoothing_window = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(("argument_name", "value"), INVALID_ARGUMENTS)
def test_preprocess_config_rejects_invalid_arguments(
    argument_name: str, value: object
) -> None:
    """Reject each invalid argument at construction."""
    match = "max_null_ratio must be between" if argument_name == "max_null_ratio" else argument_name
    with pytest.raises(ValueError, match=match):
        PreprocessConfig(**{argument_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("materialize_once", [True, False])
@pytest.mark.parametrize(("argument_name", "value"), INVALID_ARGUMENTS)
def test_preprocess_and_flatten_rejects_invalid_arguments_before_reading_input(
    argument_name: str, value: object, materialize_once: bool, tmp_path: Path
) -> None:
    """Report an invalid argument before touching a missing input path."""
    missing_path = tmp_path / "missing.parquet"
    match = "max_null_ratio must be between" if argument_name == "max_null_ratio" else argument_name

    with pytest.raises(ValueError, match=match):
        preprocess_and_flatten(  # type: ignore[arg-type]
            [missing_path],
            materialize_once=materialize_once,
            **{argument_name: value},
        )


def test_flatten_pca_rejects_invalid_arguments_before_reading_input(
    tmp_path: Path,
) -> None:
    """Validate preprocessing arguments at the flatten_pca entry point."""
    with pytest.raises(ValueError, match="t_smoothing_window"):
        flatten_pca([tmp_path / "missing.parquet"], t_smoothing_window=-1.0)


def test_flatten_pca_ignores_preprocessing_arguments_with_flattened(
    real_fixture_paths: list[Path],
) -> None:
    """Keep ignoring preprocessing arguments when ``flattened`` is supplied."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])

    model = flatten_pca(
        flattened=flattened,
        n_component=1,
        t_smoothing_window=-1.0,
        t_downsampling_stride=0,
    )

    assert model.pca.n_components_ == 1
