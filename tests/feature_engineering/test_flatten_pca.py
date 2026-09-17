"""End-to-end tests for the public Flatten-PCA workflow."""

from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spca.feature_engineering import flatten_pca
from spca.feature_engineering.flatten_pca import flatten_pca as package_flatten_pca
from spca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from spca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)
from spca.feature_engineering.flatten_pca.normalization import (
    apply_t_normalization as _apply_t_normalization,
)
from spca.feature_engineering.flatten_pca.normalization import (
    apply_w_normalization as _apply_w_normalization,
)
from spca.feature_engineering.flatten_pca.smoothing import (
    apply_t_smoothing as _apply_t_smoothing,
)
from spca.feature_engineering.flatten_pca.smoothing import (
    apply_w_smoothing as _apply_w_smoothing,
)
from spca.feature_engineering.pca import fit_and_transform_pca

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def test_public_import_and_signature_are_stable(
    real_fixture_paths: list[Path],
) -> None:
    """Preserve the documented public import and keyword-only API contract."""
    parameters = signature(flatten_pca).parameters

    assert flatten_pca is package_flatten_pca
    assert list(parameters) == [
        "paths",
        "n_component",
        "t_smoothing_window",
        "w_smoothing_window",
        "t_normalization_range",
        "w_normalization_range",
        "t_downsampling_stride",
        "w_downsampling_stride",
    ]
    assert parameters["paths"].kind is Parameter.POSITIONAL_OR_KEYWORD
    for parameter in list(parameters.values())[1:]:
        assert parameter.kind is Parameter.KEYWORD_ONLY
    for name in list(parameters)[1:6]:
        assert parameters[name].default is None
    assert parameters["t_downsampling_stride"].default == 1
    assert parameters["w_downsampling_stride"].default == 1

    result = flatten_pca(real_fixture_paths, n_component=1)
    assert result.height == len(real_fixture_paths)
    assert result.columns[-1] == "pca-1"


def _expected_flatten_pca(
    paths: list[Path],
    *,
    n_component: int,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Build the expected public workflow from its specified stages.

    Parameters
    ----------
    paths : list[Path]
        Real Parquet fixture paths.
    n_component : int
        Number of PCA score columns.
    t_smoothing_window : float | None, default None
        Time-direction smoothing half-window.
    w_smoothing_window : float | None, default None
        Wavelength-direction smoothing half-window.
    t_normalization_range : tuple[float, float] | None, default None
        Inclusive time normalization interval.
    w_normalization_range : tuple[float, float] | None, default None
        Inclusive wavelength normalization interval.

    Returns
    -------
    pl.DataFrame
        Flattened feature rows with appended PCA scores.
    """
    prepared_inputs = []
    for path, frame in _load_and_validate_inputs(paths):
        prepared = _apply_t_smoothing(frame, t_smoothing_window)
        prepared = _apply_w_smoothing(prepared, w_smoothing_window)
        prepared = _apply_t_normalization(prepared, t_normalization_range)
        prepared = _apply_w_normalization(prepared, w_normalization_range)
        prepared_inputs.append((path, prepared))
    flattened = _flatten_inputs(prepared_inputs)
    feature_columns = flattened.columns[1:]
    return fit_and_transform_pca(
        df=flattened.lazy(),
        columns=feature_columns,
        n_component=n_component,
        max_n_component=None,
        impute_strategy="drop",
        outlier_strategy=None,
        scaling_strategy="none",
    ).collect()


def _assert_flatten_pca_matches(
    result: pl.DataFrame,
    expected: pl.DataFrame,
    n_component: int,
) -> None:
    """Compare workflow output while permitting independent PCA sign flips.

    Parameters
    ----------
    result : pl.DataFrame
        Public API output.
    expected : pl.DataFrame
        Output assembled directly from the specified processing stages.
    n_component : int
        Number of trailing PCA score columns.
    """
    feature_columns = result.columns[1:-n_component]
    pca_columns = result.columns[-n_component:]
    assert result.columns == expected.columns
    assert result["filename"].to_list() == expected["filename"].to_list()
    assert result.select(feature_columns).to_numpy() == pytest.approx(
        expected.select(feature_columns).to_numpy()
    )
    for column in pca_columns:
        actual_scores = result[column].to_numpy()
        expected_scores = expected[column].to_numpy()
        assert np.allclose(actual_scores, expected_scores) or np.allclose(
            actual_scores,
            -expected_scores,
        )


def test_flatten_pca_runs_all_real_fixtures_end_to_end(
    real_fixture_paths: list[Path],
) -> None:
    """Return deterministic flattened features and PCA scores for all fixtures."""
    paths = real_fixture_paths
    expected = _expected_flatten_pca(paths, n_component=3)

    result = flatten_pca(paths, n_component=3)

    assert len(paths) == 6
    assert result.columns == [*expected.columns[:-3], "pca-1", "pca-2", "pca-3"]
    assert result.height == len(paths)
    _assert_flatten_pca_matches(result, expected, 3)
    assert np.isfinite(result.select(pl.exclude("filename")).to_numpy()).all()
    features = result.select(result.columns[1:-3]).to_numpy()
    scores = result.select(result.columns[-3:]).to_numpy()
    singular_values = np.linalg.svd(
        features - features.mean(axis=0),
        compute_uv=False,
    )
    assert np.square(scores).sum(axis=0) == pytest.approx(
        np.square(singular_values[:3])
    )


def test_readme_minimal_public_api_example_runs_on_all_real_fixtures(
    real_fixture_paths: list[Path],
) -> None:
    """Keep the documented optional-component real-data example executable."""
    readme = Path("README.md").read_text(encoding="utf-8")
    paths = real_fixture_paths
    fixture_frames = [pl.read_parquet(path) for path in paths]

    from spca.feature_engineering import flatten_pca as public_flatten_pca

    result = public_flatten_pca(paths)

    assert "from spca.feature_engineering import flatten_pca" in readme
    assert "flatten_pca(paths)" in readme
    assert len(paths) == len(fixture_frames) == 6
    assert result.height == len(paths)
    assert result.columns[-6:] == [f"pca-{index}" for index in range(1, 7)]


@pytest.mark.parametrize(
    "preprocessing",
    [
        {"t_smoothing_window": 0.5},
        {"w_smoothing_window": 0.5},
        {"t_normalization_range": (0.0, 4.0)},
        {"w_normalization_range": (350.0, 850.0)},
    ],
)
def test_flatten_pca_applies_each_preprocessing_stage(
    preprocessing: dict[str, object],
    real_fixture_paths: list[Path],
) -> None:
    """Match the specified pipeline when each preprocessing stage is enabled."""
    paths = real_fixture_paths
    expected = _expected_flatten_pca(paths, n_component=2, **preprocessing)  # type: ignore[arg-type]

    result = flatten_pca(paths, n_component=2, **preprocessing)  # type: ignore[arg-type]

    _assert_flatten_pca_matches(result, expected, 2)


def test_flatten_pca_applies_all_preprocessing_in_specified_order(
    real_fixture_paths: list[Path],
) -> None:
    """Apply t/w smoothing before t/w normalization using real fixture data."""
    paths = real_fixture_paths
    first = pl.read_parquet(paths[0])
    wavelengths = [
        float(column.removesuffix("nm"))
        for column in first.columns
        if column not in METADATA_COLUMNS
    ]
    preprocessing = {
        "t_smoothing_window": 0.5,
        "w_smoothing_window": 0.5,
        "t_normalization_range": (
            float(first["Time"].min()),
            float(first["Time"].max()),
        ),
        "w_normalization_range": (min(wavelengths), max(wavelengths)),
    }
    expected = _expected_flatten_pca(paths, n_component=2, **preprocessing)

    result = flatten_pca(paths, n_component=2, **preprocessing)

    _assert_flatten_pca_matches(result, expected, 2)


@pytest.mark.parametrize("n_component", [0, 7])
def test_flatten_pca_rejects_invalid_component_counts(
    n_component: int, real_fixture_paths: list[Path]
) -> None:
    """Reject component counts outside one through the PCA matrix rank bound."""
    with pytest.raises(ValueError, match="n_component"):
        flatten_pca(real_fixture_paths, n_component=n_component)
