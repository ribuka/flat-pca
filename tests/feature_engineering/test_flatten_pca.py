"""End-to-end tests for the public Flatten-PCA workflow."""

from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA

from spca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
    reshape_pca_components,
)
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
    assert isinstance(result, PCA)
    assert result.n_components_ == 1


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
    feature_columns = flattened.collect_schema().names()[1:]
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

    pca = flatten_pca(paths, n_component=3)
    result = append_pca_scores(pca, preprocess_and_flatten(paths)).collect()

    assert len(paths) == 6
    assert isinstance(pca, PCA)
    assert pca.n_components_ == 3
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
    """Keep the documented four-stage API example executable on real Parquet."""
    readme = Path("README.md").read_text(encoding="utf-8")
    paths = real_fixture_paths
    fixture_frames = [pl.read_parquet(path) for path in paths]

    from spca.feature_engineering import (
        append_pca_scores as public_append_pca_scores,
    )
    from spca.feature_engineering import flatten_pca as public_flatten_pca
    from spca.feature_engineering import (
        preprocess_and_flatten as public_preprocess_and_flatten,
    )
    from spca.feature_engineering import (
        reshape_pca_components as public_reshape_pca_components,
    )

    flattened = public_preprocess_and_flatten(paths)
    pca = public_flatten_pca(paths, n_component=2)
    result = public_append_pca_scores(pca, flattened).collect()
    components = public_reshape_pca_components(pca, flattened)

    for public_name in (
        "preprocess_and_flatten",
        "flatten_pca",
        "append_pca_scores",
        "reshape_pca_components",
    ):
        assert public_name in readme
    assert len(paths) == len(fixture_frames) == 6
    assert isinstance(flattened, pl.LazyFrame)
    assert isinstance(pca, PCA)
    assert result.columns[-2:] == ["pca-1", "pca-2"]
    assert components.shape[0] == pca.n_components_ == 2


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

    result = append_pca_scores(
        flatten_pca(paths, n_component=2, **preprocessing),  # type: ignore[arg-type]
        preprocess_and_flatten(paths, **preprocessing),  # type: ignore[arg-type]
    ).collect()

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

    result = append_pca_scores(
        flatten_pca(paths, n_component=2, **preprocessing),
        preprocess_and_flatten(paths, **preprocessing),
    ).collect()

    _assert_flatten_pca_matches(result, expected, 2)


@pytest.mark.parametrize("n_component", [0, 7, True, 1.5])
def test_flatten_pca_rejects_invalid_component_counts(
    n_component: object, real_fixture_paths: list[Path]
) -> None:
    """Reject component counts outside one through the PCA matrix rank bound."""
    with pytest.raises(ValueError, match="n_component"):
        flatten_pca(real_fixture_paths, n_component=n_component)


def test_append_pca_scores_preserves_real_flattened_rows_and_columns(
    real_fixture_paths: list[Path],
) -> None:
    """Append finite PCA scores to real-fixture flattened rows in their input order."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    pca = flatten_pca(real_fixture_paths[:3], n_component=2)

    result = append_pca_scores(pca, flattened)
    materialized_flattened = flattened.collect()
    materialized_result = result.collect()
    feature_columns = materialized_flattened.columns[1:]

    assert isinstance(result, pl.LazyFrame)
    assert materialized_result.columns == [
        "filename",
        *feature_columns,
        "pca-1",
        "pca-2",
    ]
    assert materialized_result["filename"].to_list() == materialized_flattened[
        "filename"
    ].to_list()
    scores = materialized_result.select(["pca-1", "pca-2"]).to_numpy()
    assert np.isfinite(scores).all()
    assert scores == pytest.approx(
        pca.transform(materialized_flattened.select(feature_columns).to_numpy())
    )
    reconstructed = scores @ pca.components_ + pca.mean_
    assert reconstructed == pytest.approx(
        materialized_flattened.select(feature_columns).to_numpy()
    )


def test_append_pca_scores_rejects_feature_count_mismatch(
    real_fixture_paths: list[Path],
) -> None:
    """Reject flattened real-fixture features that do not match the fitted PCA."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    pca = flatten_pca(real_fixture_paths[:3], n_component=2)
    final_feature = flattened.collect_schema().names()[-1]

    with pytest.raises(ValueError, match="feature count"):
        append_pca_scores(pca, flattened.drop(final_feature))


def test_reshape_pca_components_places_real_flattened_features_on_sorted_axes(
    real_fixture_paths: list[Path],
) -> None:
    """Reshape real-Parquet PCA coefficients using sorted spectral coordinates."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    materialized = flattened.collect()
    pca = flatten_pca(real_fixture_paths[:3], n_component=2)

    reshaped = reshape_pca_components(pca, flattened)
    feature_columns = materialized.columns[1:]
    coordinates = [
        (
            float(parts[0].removesuffix("nm")),
            int(parts[1]),
            int(parts[2]),
            float(parts[3]),
        )
        for column in feature_columns
        for parts in [column.split("_")]
    ]
    wavelengths = sorted({coordinate[0] for coordinate in coordinates})
    steps = sorted({coordinate[1] for coordinate in coordinates})
    sequences = sorted({coordinate[2] for coordinate in coordinates})
    times = sorted({coordinate[3] for coordinate in coordinates})

    assert reshaped.shape == (2, len(wavelengths), len(steps), len(sequences), len(times))
    assert reshaped.reshape(2, -1) == pytest.approx(pca.components_)
    coordinate = coordinates[0]
    assert reshaped[
        :,
        wavelengths.index(coordinate[0]),
        steps.index(coordinate[1]),
        sequences.index(coordinate[2]),
        times.index(coordinate[3]),
    ] == pytest.approx(pca.components_[:, 0])


def test_reshape_pca_components_rejects_invalid_real_flattened_layouts(
    real_fixture_paths: list[Path],
) -> None:
    """Reject product gaps, ambiguous names, and PCA mismatches from real data."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    feature_columns = flattened.collect_schema().names()[1:]
    pca = flatten_pca(real_fixture_paths[:3], n_component=2)

    with pytest.raises(ValueError, match="feature count"):
        reshape_pca_components(pca, flattened.drop(feature_columns[-1]))
    with pytest.raises(ValueError, match="uniquely decode"):
        reshape_pca_components(
            pca,
            flattened.rename({feature_columns[0]: "ambiguous_feature"}),
        )

    incomplete = flattened.drop(feature_columns[-1])
    incomplete_features = incomplete.collect_schema().names()[1:]
    incomplete_pca = PCA(n_components=1).fit(
        incomplete.collect().select(incomplete_features).to_numpy()
    )
    with pytest.raises(ValueError, match="Cartesian product"):
        reshape_pca_components(incomplete_pca, incomplete)


def test_preprocess_and_flatten_returns_lazy_real_fixture_query(
    real_fixture_paths: list[Path],
) -> None:
    """Defer real-Parquet preprocessing and match the established flatten contract."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    expected_inputs = [
        (path, pl.read_parquet(path)) for path in sorted(real_fixture_paths[:3])
    ]
    expected = _flatten_inputs(expected_inputs)

    assert isinstance(flattened, pl.LazyFrame)
    assert isinstance(expected, pl.DataFrame)
    assert flattened.collect().equals(expected)


@pytest.mark.parametrize(
    "preprocessing",
    [
        {"t_smoothing_window": 0.5},
        {"w_smoothing_window": 0.5},
        {"t_normalization_range": (0.0, 4.0)},
        {"w_normalization_range": (350.0, 850.0)},
        {
            "t_smoothing_window": 0.5,
            "w_smoothing_window": 0.5,
            "t_normalization_range": (0.0, 4.0),
            "w_normalization_range": (350.0, 850.0),
            "t_downsampling_stride": 2,
            "w_downsampling_stride": 2,
        },
    ],
)
def test_preprocess_and_flatten_matches_eager_stage_contract(
    preprocessing: dict[str, object], real_fixture_paths: list[Path]
) -> None:
    """Match eager real-fixture stages for each preprocessing configuration."""
    loaded = [(path, pl.read_parquet(path)) for path in real_fixture_paths[:3]]
    frames = [frame for _, frame in loaded]
    from spca.feature_engineering.flatten_pca.downsampling import (
        apply_t_downsampling,
        apply_w_downsampling,
        collect_unique_times,
        collect_unique_wavelengths,
    )

    prepared = []
    for path, frame in loaded:
        frame = _apply_t_smoothing(frame, preprocessing.get("t_smoothing_window"))
        frame = _apply_w_smoothing(frame, preprocessing.get("w_smoothing_window"))
        frame = _apply_t_normalization(frame, preprocessing.get("t_normalization_range"))
        frame = _apply_w_normalization(frame, preprocessing.get("w_normalization_range"))
        frame = apply_t_downsampling(
            frame, collect_unique_times(frames), preprocessing.get("t_downsampling_stride", 1)
        )
        frame = apply_w_downsampling(
            frame, collect_unique_wavelengths(frames), preprocessing.get("w_downsampling_stride", 1)
        )
        assert isinstance(frame, pl.DataFrame)
        prepared.append((path, frame))
    expected = _flatten_inputs(prepared)
    actual = preprocess_and_flatten(
        list(reversed(real_fixture_paths[:3])), **preprocessing  # type: ignore[arg-type]
    ).collect()

    assert isinstance(expected, pl.DataFrame)
    assert actual.equals(expected)


def test_preprocess_and_flatten_rejects_real_fixture_schema_variant(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Preserve ValueError validation for a real-fixture-derived bad input."""
    invalid_path = tmp_path / "missing-time.parquet"
    pl.read_parquet(real_fixture_paths[0]).drop("Time").write_parquet(invalid_path)

    with pytest.raises(ValueError, match="required columns"):
        preprocess_and_flatten([invalid_path])
