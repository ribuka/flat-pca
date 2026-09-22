"""End-to-end tests for the public Flatten-PCA workflow."""

import time
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
    reshape_pca_components,
)
from flat_pca.feature_engineering.flatten_pca import api as _api
from flat_pca.feature_engineering.flatten_pca import flatten_pca as package_flatten_pca
from flat_pca.feature_engineering.flatten_pca import pca_scores
from flat_pca.feature_engineering.flatten_pca.flatten import (
    flatten_inputs as _flatten_inputs,
)
from flat_pca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)
from flat_pca.feature_engineering.pca import PcaModel, fit_and_transform_pca
from flat_pca.feature_engineering.preprocess import (
    add_step_time_columns as _add_step_time_columns,
)
from flat_pca.feature_engineering.preprocess import smoothing as _smoothing
from flat_pca.feature_engineering.preprocess.normalization import (
    apply_t_normalization as _apply_t_normalization,
)
from flat_pca.feature_engineering.preprocess.normalization import (
    apply_w_normalization as _apply_w_normalization,
)
from flat_pca.feature_engineering.preprocess.smoothing import (
    apply_t_smoothing as _apply_t_smoothing,
)
from flat_pca.feature_engineering.preprocess.smoothing import (
    apply_w_smoothing as _apply_w_smoothing,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def test_public_import_and_signature_are_stable(
    real_fixture_paths: list[Path],
) -> None:
    """Preserve the documented public import and keyword-only API contract."""
    parameters = signature(flatten_pca).parameters

    assert flatten_pca is package_flatten_pca
    assert list(parameters) == [
        "paths",
        "flattened",
        "n_component",
        "target_steps",
        "edge_trim",
        "wavelength_range",
        "t_smoothing_window",
        "w_smoothing_window",
        "t_normalization_range",
        "w_normalization_range",
        "t_downsampling_stride",
        "w_downsampling_stride",
        "max_null_ratio",
        "stem_uniqueness",
        "validate_metadata_uniqueness",
        "validate_metadata_alignment",
        "materialize_once",
        "impute_strategy",
        "impute_kmeans_n_clusters",
    ]
    assert parameters["paths"].kind is Parameter.POSITIONAL_OR_KEYWORD
    for parameter in list(parameters.values())[1:]:
        assert parameter.kind is Parameter.KEYWORD_ONLY
    for name in list(parameters)[0:10]:
        assert parameters[name].default is None
    assert parameters["t_downsampling_stride"].default == 1
    assert parameters["w_downsampling_stride"].default == 1
    assert parameters["max_null_ratio"].default == 0.1
    assert parameters["stem_uniqueness"].default == "skip"
    assert parameters["validate_metadata_uniqueness"].default is False
    assert parameters["validate_metadata_alignment"].default is False
    assert parameters["materialize_once"].default is True
    assert parameters["impute_strategy"].default == "drop"
    assert parameters["impute_kmeans_n_clusters"].default is None

    result = flatten_pca(real_fixture_paths, n_component=1)
    assert isinstance(result, PcaModel)
    assert result.pca.n_components_ == 1


def test_flatten_pca_accepts_exactly_one_real_fixture_input_source(
    real_fixture_paths: list[Path],
) -> None:
    """Fit identical PcaModels from paths or an already flattened real query."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])

    from_paths = flatten_pca(real_fixture_paths[:3], n_component=2)
    from_flattened = flatten_pca(flattened=flattened, n_component=2)

    assert isinstance(from_paths, PcaModel)
    assert isinstance(from_flattened, PcaModel)
    assert from_paths.columns == from_flattened.columns
    assert from_paths.pca.components_ == pytest.approx(from_flattened.pca.components_)
    assert from_paths.pca.explained_variance_ == pytest.approx(
        from_flattened.pca.explained_variance_
    )
    with pytest.raises(ValueError, match="exactly one"):
        flatten_pca(n_component=2)
    with pytest.raises(ValueError, match="exactly one"):
        flatten_pca(real_fixture_paths[:3], flattened=flattened, n_component=2)


def test_public_api_optionally_rejects_duplicate_metadata_keys(
    tmp_path: Path,
    real_fixture_paths: list[Path],
) -> None:
    """Forward duplicate-key validation through both public path APIs."""
    frame = pl.read_parquet(real_fixture_paths[0])
    path = tmp_path / "duplicate-key.parquet"
    pl.concat([frame, frame.head(1)]).write_parquet(path)

    with pytest.raises(ValueError, match="duplicate metadata keys"):
        preprocess_and_flatten([path], validate_metadata_uniqueness=True)
    with pytest.raises(ValueError, match="duplicate metadata keys"):
        flatten_pca([path], n_component=1, validate_metadata_uniqueness=True)


def test_public_api_allows_duplicate_metadata_keys_by_default(
    tmp_path: Path,
    real_fixture_paths: list[Path],
) -> None:
    """Skip duplicate-key validation through public APIs by default."""
    frame = pl.read_parquet(real_fixture_paths[0])
    path = tmp_path / "duplicate-key.parquet"
    pl.concat([frame, frame.head(1)]).write_parquet(path)

    flattened = preprocess_and_flatten([path])
    pca = flatten_pca([path, real_fixture_paths[1]], n_component=1)

    assert isinstance(flattened, pl.LazyFrame)
    assert flattened.collect().height == 1
    assert isinstance(pca, PcaModel)


def test_fit_flattened_pca_delegates_to_common_pca_configuration(
    real_fixture_paths: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the shared PCA fitter with the required unmodified-data settings."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    original_fit_pca = pca_scores.fit_pca
    captured: dict[str, object] = {}

    def capture_fit_pca(**kwargs: object) -> PcaModel:
        """Capture delegated PCA-fitting arguments while retaining behavior."""
        captured.update(kwargs)
        return original_fit_pca(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pca_scores, "fit_pca", capture_fit_pca)
    model = pca_scores.fit_flattened_pca(flattened, n_component=2)

    assert isinstance(model, PcaModel)
    assert captured["df"] is flattened
    assert captured["columns"] == flattened.collect_schema().names()[1:]
    assert captured["n_component"] == 2
    assert captured["impute_strategy"] == "drop"
    assert captured["outlier_strategy"] is None
    assert captured["scaling_strategy"] == "none"
    assert captured["max_n_component"] is None


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
        frame = _add_step_time_columns(frame)
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
    assert result["source"].to_list() == expected["source"].to_list()
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
    assert isinstance(pca, PcaModel)
    assert pca.pca.n_components_ == 3
    assert result.columns == [*expected.columns[:-3], "pca-1", "pca-2", "pca-3"]
    assert result.height == len(paths)
    _assert_flatten_pca_matches(result, expected, 3)
    assert np.isfinite(result.select(pl.exclude("source")).to_numpy()).all()
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

    from flat_pca.feature_engineering import (
        append_pca_scores as public_append_pca_scores,
    )
    from flat_pca.feature_engineering import flatten_pca as public_flatten_pca
    from flat_pca.feature_engineering import (
        preprocess_and_flatten as public_preprocess_and_flatten,
    )
    from flat_pca.feature_engineering import (
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
    assert isinstance(pca, PcaModel)
    assert result.columns[-2:] == ["pca-1", "pca-2"]
    assert components["component"].n_unique() == pca.pca.n_components_ == 2


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
        "source",
        *feature_columns,
        "pca-1",
        "pca-2",
    ]
    assert materialized_result["source"].to_list() == materialized_flattened[
        "source"
    ].to_list()
    scores = materialized_result.select(["pca-1", "pca-2"]).to_numpy()
    assert np.isfinite(scores).all()
    assert scores == pytest.approx(
        pca.pca.transform(materialized_flattened.select(feature_columns).to_numpy())
    )
    reconstructed = scores @ pca.pca.components_ + pca.pca.mean_
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


def _build_missing_combo_flattened(real_fixture_paths: list[Path]) -> pl.LazyFrame:
    """Flatten three real fixtures where one is missing its last StepTime combo.

    Bypasses ``preprocess_and_flatten``'s ``max_null_ratio`` column pruning
    (by calling ``flatten_inputs`` directly, as
    ``tests/feature_engineering/test_flatten_pca_flatten.py``'s
    ``_build_union_gap_fixtures`` does) so the null cells produced by
    flatten's normal cross-file union-and-null-fill behavior survive into the
    returned frame, reproducing the missing values that reach
    ``append_pca_scores`` in real usage. Two files stay complete so that
    dropping the incomplete one still leaves more than one sample to fit.
    """
    path_a, path_b, path_c = real_fixture_paths[:3]
    frame_a = _add_step_time_columns(_load_and_validate_inputs([path_a])[0][1]).collect()
    frame_b = _add_step_time_columns(_load_and_validate_inputs([path_b])[0][1]).collect()
    frame_c_full = _add_step_time_columns(
        _load_and_validate_inputs([path_c])[0][1]
    ).collect()
    frame_c = frame_c_full.slice(0, frame_c_full.height - 1)
    flattened = _flatten_inputs(
        [(path_a, frame_a.lazy()), (path_b, frame_b.lazy()), (path_c, frame_c.lazy())]
    )
    assert isinstance(flattened, pl.LazyFrame)
    return flattened


def test_append_pca_scores_drops_rows_with_remaining_missing_values(
    real_fixture_paths: list[Path],
) -> None:
    """Regression test: flatten's union nulls no longer crash append_pca_scores.

    Before the fix, ``append_pca_scores`` passed ``flattened``'s raw values
    (including nulls left by ``flatten_inputs``'s cross-file union) directly
    into ``sklearn``'s ``PCA.transform``, which raises ``ValueError: Input X
    contains NaN``. Delegating to ``transform_pca`` applies the fitted
    ``impute_strategy`` first, so with the default ``"drop"`` strategy the
    row with remaining nulls is excluded instead of crashing.
    """
    flattened = _build_missing_combo_flattened(real_fixture_paths)
    pca = flatten_pca(flattened=flattened, n_component=1, impute_strategy="drop")

    result = append_pca_scores(pca, flattened).collect()

    assert result.height == 2
    assert result["source"].to_list() == [
        real_fixture_paths[0].as_posix(),
        real_fixture_paths[1].as_posix(),
    ]
    assert np.isfinite(result.select(pl.exclude("source")).to_numpy()).all()


def test_append_pca_scores_fills_remaining_missing_values_with_median(
    real_fixture_paths: list[Path],
) -> None:
    """Fill remaining missing values with the fitted median instead of dropping rows."""
    flattened = _build_missing_combo_flattened(real_fixture_paths)
    pca = flatten_pca(flattened=flattened, n_component=1, impute_strategy="median")

    result = append_pca_scores(pca, flattened).collect()

    assert result.height == 3
    assert np.isfinite(result.select(pl.exclude("source")).to_numpy()).all()


def test_append_pca_scores_fills_remaining_missing_values_with_kmeans(
    real_fixture_paths: list[Path],
) -> None:
    """Fill remaining missing values from the nearest fitted cluster centroid."""
    flattened = _build_missing_combo_flattened(real_fixture_paths)
    pca = flatten_pca(
        flattened=flattened,
        n_component=1,
        impute_strategy="kmeans",
        impute_kmeans_n_clusters=2,
    )

    result = append_pca_scores(pca, flattened).collect()

    assert result.height == 3
    assert np.isfinite(result.select(pl.exclude("source")).to_numpy()).all()
    assert result["source"].to_list() == flattened.collect()["source"].to_list()


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

    assert reshaped.columns == [
        "StepTime", "Step", "Sequence", "wavelength", "component", "coefficient"
    ]
    assert reshaped.height == 2 * len(wavelengths) * len(steps) * len(sequences) * len(times)
    assert reshaped.equals(
        reshaped.sort(["Step", "Sequence", "StepTime", "component", "wavelength"])
    )

    coordinate_frame = pl.DataFrame(
        {
            "wavelength": [coordinate[0] for coordinate in coordinates],
            "Step": [coordinate[1] for coordinate in coordinates],
            "Sequence": [coordinate[2] for coordinate in coordinates],
            "StepTime": [coordinate[3] for coordinate in coordinates],
        }
    )
    sort_keys = ["wavelength", "Step", "Sequence", "StepTime"]
    for component_index in range(pca.pca.n_components_):
        expected = coordinate_frame.with_columns(
            pl.Series("coefficient", pca.pca.components_[component_index, :])
        ).sort(sort_keys)
        actual = (
            reshaped.filter(pl.col("component") == component_index + 1)
            .select(["wavelength", "Step", "Sequence", "StepTime", "coefficient"])
            .sort(sort_keys)
        )
        assert actual["coefficient"].to_numpy() == pytest.approx(
            expected["coefficient"].to_numpy()
        )


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
    incomplete_pca = pca_scores.fit_flattened_pca(incomplete, 1)
    with pytest.raises(ValueError, match="Cartesian product"):
        reshape_pca_components(incomplete_pca, incomplete)


def test_preprocess_and_flatten_returns_lazy_real_fixture_query(
    real_fixture_paths: list[Path],
) -> None:
    """Defer real-Parquet preprocessing and match the established flatten contract."""
    flattened = preprocess_and_flatten(real_fixture_paths[:3])
    expected_inputs = [
        (path.resolve(), _add_step_time_columns(pl.read_parquet(path)))
        for path in sorted(real_fixture_paths[:3])
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
    from flat_pca.feature_engineering.preprocess.downsampling import (
        apply_t_downsampling,
        apply_w_downsampling,
        collect_unique_times,
        collect_unique_wavelengths,
    )

    prepared = []
    for path, frame in loaded:
        frame = _add_step_time_columns(frame)
        assert isinstance(frame, pl.DataFrame)
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
        prepared.append((path.resolve(), frame))
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


def _write_extra_step_variant(
    frame: pl.DataFrame, baseline_path: Path, variant_path: Path
) -> None:
    """Write a baseline real-fixture frame and a variant with an extra Step row.

    Parameters
    ----------
    frame : pl.DataFrame
        Real-fixture-derived data shared by both written files.
    baseline_path : Path
        Destination for the unmodified frame.
    variant_path : Path
        Destination for the frame plus one row at a ``Step`` and ``Time``
        combination absent from ``baseline_path``.
    """
    frame.write_parquet(baseline_path)
    extra_step_row = frame.head(1).with_columns(
        pl.lit(2).alias("Step"),
        pl.lit(frame["Time"].max() + 100.0).alias("Time"),
    )
    pl.concat([frame, extra_step_row]).write_parquet(variant_path)


def test_preprocess_and_flatten_skips_metadata_alignment_check_by_default(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Flatten mismatched real-fixture-derived inputs by default, null-filling gaps."""
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = tmp_path / "baseline.parquet"
    variant = tmp_path / "variant.parquet"
    _write_extra_step_variant(frame, baseline, variant)
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    extra_sequence = int(frame.head(1)["Sequence"].item())
    extra_column = f"{wavelength}_2_{extra_sequence}_0.00"

    # max_null_ratio=1.0 disables sparse-column pruning so this test isolates
    # the union/null-fill behavior itself; see
    # test_preprocess_and_flatten_prunes_sparse_columns_by_default for the
    # default-threshold pruning behavior.
    flattened = preprocess_and_flatten(
        [baseline, variant], max_null_ratio=1.0
    ).collect()
    row_by_source = {row["source"]: row for row in flattened.iter_rows(named=True)}

    assert flattened.height == 2
    assert extra_column in flattened.columns
    assert row_by_source[baseline.resolve().as_posix()][extra_column] is None
    assert row_by_source[variant.resolve().as_posix()][extra_column] is not None


def test_preprocess_and_flatten_prunes_sparse_columns_by_default(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Drop a 50%-missing column under the default max_null_ratio threshold."""
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = tmp_path / "baseline.parquet"
    variant = tmp_path / "variant.parquet"
    _write_extra_step_variant(frame, baseline, variant)
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    extra_sequence = int(frame.head(1)["Sequence"].item())
    extra_column = f"{wavelength}_2_{extra_sequence}_0.00"

    flattened = preprocess_and_flatten([baseline, variant]).collect()

    assert flattened.height == 2
    assert extra_column not in flattened.columns


def test_flatten_pca_survives_per_file_unique_metadata_gaps(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Fit PCA even when every file has its own unique missing metadata combo.

    Reproduces the PR #4 review concern: with impute_strategy="drop" (the
    default), any column with a null drops that entire row. Before
    sparse-column pruning, giving each of 3 files its own unique missing
    (Step, Sequence, StepTime) combination meant every row had at least one
    null feature, so every row was dropped and PCA fitting raised "no rows
    remain after missing-value handling". Sparse-column pruning removes
    those per-file-unique columns (each 33% missing, above the default 0.1
    threshold) before PCA fitting, so no nulls remain and all 3 rows
    survive.
    """
    source_frames = [pl.read_parquet(path) for path in real_fixture_paths[:3]]
    tail_start = source_frames[0].height - 3
    paths = []
    for offset, source_frame in enumerate(source_frames):
        # Always keep row 0 (the minimum Time) so add_step_time_columns's
        # per-segment StepTime origin stays identical across all 3 files;
        # only ever drop a row from the tail. Using 3 distinct real
        # fixtures (rather than 3 copies of one) keeps genuine spectral
        # variance across samples for PCA to fit.
        missing_index = tail_start + offset
        trimmed = source_frame.filter(pl.int_range(pl.len()) != missing_index)
        path = tmp_path / f"gap_{offset}.parquet"
        trimmed.write_parquet(path)
        paths.append(path)

    pca = flatten_pca(paths, n_component=1)

    assert isinstance(pca, PcaModel)
    assert pca.pca.n_components_ == 1


def test_preprocess_and_flatten_applies_target_steps_before_metadata_alignment_check(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Filter Step rows before the opt-in Time/Step/Sequence alignment check."""
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = tmp_path / "baseline.parquet"
    variant = tmp_path / "variant.parquet"
    _write_extra_step_variant(frame, baseline, variant)

    with pytest.raises(ValueError, match="metadata-key sets"):
        preprocess_and_flatten(
            [baseline, variant], validate_metadata_alignment=True
        )

    flattened = preprocess_and_flatten(
        [baseline, variant],
        target_steps=[1],
        validate_metadata_alignment=True,
    ).collect()

    assert flattened.height == 2


def test_preprocess_and_flatten_materialize_once_matches_deferred_result(
    real_fixture_paths: list[Path],
) -> None:
    """Return an equal `pl.LazyFrame` result whether materialize_once is True or False."""
    paths = real_fixture_paths[:3]

    materialized = preprocess_and_flatten(paths)
    default = preprocess_and_flatten(paths, materialize_once=True)
    deferred = preprocess_and_flatten(paths, materialize_once=False)

    assert isinstance(materialized, pl.LazyFrame)
    assert isinstance(deferred, pl.LazyFrame)
    assert materialized.collect().equals(default.collect())
    assert materialized.collect().equals(deferred.collect())


def _instrument_flatten_execution_count(
    monkeypatch: pytest.MonkeyPatch,
) -> list[None]:
    """Wrap real flatten queries with a list recording each execution batch.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the API module's flatten-query constructor.

    Returns
    -------
    list[None]
        One item for each executed flattened query batch. This test uses the
        pinned Polars non-streaming engine, where the fixture query executes
        as one batch; it deliberately counts batches rather than a public
        Polars execution counter.
    """
    executions: list[None] = []
    original_flatten_inputs = _api.flatten_inputs

    def count_flatten_inputs(
        inputs: list[tuple[Path, pl.DataFrame]] | list[tuple[Path, pl.LazyFrame]],
    ) -> pl.DataFrame | pl.LazyFrame:
        """Wrap the real flatten query with an execution counter."""
        flattened = original_flatten_inputs(inputs)
        if isinstance(flattened, pl.DataFrame):
            # preprocess_and_flatten's materialize_once=True branch passes
            # already-collected DataFrame inputs, so flatten_inputs executes
            # synchronously here rather than on a later .collect().
            executions.append(None)
            return flattened

        def count_batch(batch: pl.DataFrame) -> pl.DataFrame:
            """Record one executed input batch without altering its data."""
            executions.append(None)
            return batch

        return flattened.map_batches(
            count_batch,
            schema=flattened.collect_schema(),
        )

    monkeypatch.setattr(_api, "flatten_inputs", count_flatten_inputs)
    return executions


def test_preprocess_and_flatten_materializes_upstream_flatten_once(
    real_fixture_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialize real-Parquet flatten inputs once before sparse pruning."""
    executions = _instrument_flatten_execution_count(monkeypatch)

    flattened = _api.preprocess_and_flatten(real_fixture_paths[:3])

    assert len(executions) == 1
    assert flattened.collect().height == 3
    assert len(executions) == 1


def _instrument_t_smoothing_execution_count(
    monkeypatch: pytest.MonkeyPatch,
) -> list[None]:
    """Wrap the real Time-smoothing stage with a list recording each execution.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the smoothing module's eager implementation.

    Returns
    -------
    list[None]
        One item for each executed Time-smoothing batch. ``apply_t_smoothing``
        defers to this implementation through ``map_batches``, so the item
        count equals how many times the upstream lazy query actually ran.
    """
    executions: list[None] = []
    original_apply_t_smoothing_eager = _smoothing._apply_t_smoothing_eager

    def count_apply_t_smoothing_eager(frame: pl.DataFrame, window: float) -> pl.DataFrame:
        """Record one executed smoothing batch without altering its data."""
        executions.append(None)
        return original_apply_t_smoothing_eager(frame, window)

    monkeypatch.setattr(
        _smoothing, "_apply_t_smoothing_eager", count_apply_t_smoothing_eager
    )
    return executions


@pytest.mark.parametrize("materialize_once", [True, False])
def test_preprocess_and_flatten_runs_upstream_once_with_normalization(
    real_fixture_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    materialize_once: bool,
) -> None:
    """Keep the upstream execution count unchanged when normalization is enabled.

    Normalization used to ``collect()`` inside its LazyFrame branch purely to
    validate reference means, which re-ran every preceding stage on the later
    ``collect()``.
    """
    paths = real_fixture_paths[:3]

    disabled = _instrument_t_smoothing_execution_count(monkeypatch)
    preprocess_and_flatten(
        paths,
        t_smoothing_window=0.5,
        materialize_once=materialize_once,
    ).collect()
    disabled_executions = len(disabled)

    enabled = _instrument_t_smoothing_execution_count(monkeypatch)
    preprocess_and_flatten(
        paths,
        t_smoothing_window=0.5,
        t_normalization_range=(0.0, 4.0),
        w_normalization_range=(350.0, 850.0),
        materialize_once=materialize_once,
    ).collect()

    # Normalization collects each file once and reuses that result, so the
    # upstream pipeline runs exactly once per input and never more often than
    # with normalization disabled. The disabled baseline itself differs
    # between the materialize_once branches, since the deferred branch lets
    # the flatten stage rescan each per-file query.
    assert disabled_executions > 0
    assert len(enabled) == len(paths)
    assert len(enabled) <= disabled_executions


def test_preprocess_and_flatten_stays_fast_with_many_wavelengths_and_combos(
    tmp_path: Path,
) -> None:
    """Regression test guarding against the slow per-cell lazy flatten path.

    Fully synthetic data: this must exercise a (wavelength count) x (unique
    StepTime combo count) x (file count) scale far beyond what the real
    fixture (6 files x 8 rows x 16 wavelength columns) can produce, since
    only a combinatorial scale this large exposes the O(wavelengths x combos
    x files) per-cell expression blowup this test guards against. Before the
    fix, materialize_once=True (the default) always routed through that slow
    LazyFrame flatten branch regardless of scale.
    """
    n_files = 30
    n_wavelengths = 150
    n_step_time = 400
    rng = np.random.default_rng(0)
    paths = []
    for file_index in range(n_files):
        data = {
            "Time": np.arange(n_step_time, dtype=float),
            "Step": np.ones(n_step_time, dtype=int),
            "Sequence": np.ones(n_step_time, dtype=int),
        }
        for wavelength_index in range(n_wavelengths):
            data[f"{649.9 + wavelength_index:.1f}nm"] = rng.random(n_step_time)
        path = tmp_path / f"synthetic-{file_index}.parquet"
        pl.DataFrame(data).write_parquet(path)
        paths.append(path)

    start = time.perf_counter()
    flattened = preprocess_and_flatten(paths, max_null_ratio=1.0).collect()
    elapsed = time.perf_counter() - start

    assert flattened.height == n_files
    assert flattened.width == 1 + n_wavelengths * n_step_time
    assert elapsed < 30.0


def test_preprocess_and_flatten_rejects_invalid_null_ratio_before_materializing(
    real_fixture_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid thresholds before executing real-Parquet flatten inputs."""
    executions = _instrument_flatten_execution_count(monkeypatch)

    with pytest.raises(ValueError, match="threshold must be between"):
        _api.preprocess_and_flatten(real_fixture_paths[:3], max_null_ratio=1.5)

    assert executions == []


def test_materialize_once_default_decouples_result_from_source_files(
    real_fixture_paths: list[Path], tmp_path: Path
) -> None:
    """Materializing once lets later collects survive removal of the source files."""
    copied_paths = []
    for source in real_fixture_paths[:3]:
        destination = tmp_path / source.name
        destination.write_bytes(source.read_bytes())
        copied_paths.append(destination)

    materialized = preprocess_and_flatten(copied_paths)
    deferred = preprocess_and_flatten(copied_paths, materialize_once=False)
    expected = materialized.collect()

    for path in copied_paths:
        path.unlink()

    assert materialized.collect().equals(expected)
    with pytest.raises(FileNotFoundError):
        deferred.collect()


def test_flatten_pca_materialize_once_matches_deferred_pca_and_scores(
    real_fixture_paths: list[Path],
) -> None:
    """Match fitted PCA and appended scores whether materialize_once is True or False."""
    paths = real_fixture_paths[:3]

    materialized_flattened = preprocess_and_flatten(paths)
    deferred_flattened = preprocess_and_flatten(paths, materialize_once=False)

    pca_materialized = flatten_pca(flattened=materialized_flattened, n_component=2)
    pca_deferred = flatten_pca(flattened=deferred_flattened, n_component=2)
    pca_via_paths = flatten_pca(paths, n_component=2, materialize_once=False)

    result_materialized = append_pca_scores(
        pca_materialized, materialized_flattened
    ).collect()
    result_deferred = append_pca_scores(pca_deferred, deferred_flattened).collect()

    assert pca_materialized.columns == pca_deferred.columns == pca_via_paths.columns
    assert pca_materialized.pca.components_ == pytest.approx(
        pca_deferred.pca.components_
    )
    assert result_materialized.equals(result_deferred)
