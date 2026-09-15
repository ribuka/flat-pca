"""Tests for Flatten-PCA input loading and validation."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spca.feature_engineering import flatten_pca
from spca.feature_engineering.flatten_pca import (
    _apply_t_normalization,
    _apply_t_smoothing,
    _apply_w_normalization,
    _apply_w_smoothing,
    _flatten_inputs,
    _load_and_validate_inputs,
)
from spca.feature_engineering.pca import fit_and_transform_pca

FIXTURE_DIRECTORY = Path("tests/fixtures/real_subset")
METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def _fixture_paths() -> list[Path]:
    """Return the real Parquet fixture paths in filename order."""
    return sorted(FIXTURE_DIRECTORY.glob("*.parquet"), key=lambda path: path.name)


def _write_variant(frame: pl.DataFrame, path: Path) -> Path:
    """Write an in-memory real-fixture variant to pytest's temporary directory.

    Parameters
    ----------
    frame : pl.DataFrame
        Data derived from a real fixture.
    path : Path
        Temporary Parquet output path.

    Returns
    -------
    Path
        The written Parquet path.
    """
    frame.write_parquet(path)
    return path


def test_loads_one_and_multiple_real_parquet_files_deterministically() -> None:
    """Load real fixture Parquet files in normalized path order."""
    paths = _fixture_paths()
    fixture_frame = pl.read_parquet(paths[0])

    single = _load_and_validate_inputs([paths[0]])
    forward = _load_and_validate_inputs(paths[:3])
    reverse = _load_and_validate_inputs(list(reversed(paths[:3])))

    assert single[0][1].equals(fixture_frame)
    assert [path for path, _ in forward] == [path for path, _ in reverse]
    assert [path for path, _ in forward] == sorted(path.resolve() for path in paths[:3])


def test_rejects_empty_and_missing_paths(tmp_path: Path) -> None:
    """Reject an empty path collection and a nonexistent Parquet path."""
    with pytest.raises(ValueError, match="at least one"):
        _load_and_validate_inputs([])

    with pytest.raises(FileNotFoundError):
        _load_and_validate_inputs([tmp_path / "missing.parquet"])


def test_rejects_duplicate_path_stems(tmp_path: Path) -> None:
    """Reject duplicate stems even when normalized paths differ."""
    frame = pl.read_parquet(_fixture_paths()[0])
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first = _write_variant(frame, first_directory / "sample.parquet")
    second = _write_variant(frame, second_directory / "sample.parquet")

    with pytest.raises(ValueError, match="path stems"):
        _load_and_validate_inputs([first, second])


@pytest.mark.parametrize("column", ["Time", "Step", "Sequence"])
def test_rejects_missing_required_columns(tmp_path: Path, column: str) -> None:
    """Reject each missing metadata column in a real-fixture variant."""
    frame = pl.read_parquet(_fixture_paths()[0]).drop(column)
    path = _write_variant(frame, tmp_path / f"missing-{column}.parquet")

    with pytest.raises(ValueError, match="required columns"):
        _load_and_validate_inputs([path])


def test_rejects_nonnumeric_metadata_and_spectra(tmp_path: Path) -> None:
    """Reject nonnumeric metadata and spectral intensity columns."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength = next(column for column in frame.columns if column not in METADATA_COLUMNS)
    bad_metadata = _write_variant(
        frame.with_columns(pl.col("Time").cast(pl.String)),
        tmp_path / "bad-metadata.parquet",
    )
    bad_spectrum = _write_variant(
        frame.with_columns(pl.col(wavelength).cast(pl.String)),
        tmp_path / "bad-spectrum.parquet",
    )

    with pytest.raises(ValueError, match="metadata columns must be numeric"):
        _load_and_validate_inputs([bad_metadata])
    with pytest.raises(ValueError, match="spectral columns must be numeric"):
        _load_and_validate_inputs([bad_spectrum])


def test_rejects_invalid_wavelength_name(tmp_path: Path) -> None:
    """Reject a noncanonical wavelength name derived from a real fixture."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength = next(column for column in frame.columns if column not in METADATA_COLUMNS)
    path = _write_variant(
        frame.rename({wavelength: wavelength.removesuffix("nm")}),
        tmp_path / "bad-wavelength.parquet",
    )

    with pytest.raises(ValueError, match="wavelength column"):
        _load_and_validate_inputs([path])


@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf")])
def test_rejects_null_nan_and_infinite_values(
    tmp_path: Path,
    invalid_value: float | None,
) -> None:
    """Reject non-finite values inserted into a real fixture."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength = next(column for column in frame.columns if column not in METADATA_COLUMNS)
    values = frame[wavelength].to_list()
    values[0] = invalid_value
    path = _write_variant(
        frame.with_columns(pl.Series(wavelength, values)),
        tmp_path / "invalid-value.parquet",
    )

    with pytest.raises(ValueError, match="null, NaN, or infinite"):
        _load_and_validate_inputs([path])


def test_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Reject duplicate Time, Step, and Sequence tuples."""
    frame = pl.read_parquet(_fixture_paths()[0])
    path = _write_variant(
        pl.concat([frame, frame.head(1)]),
        tmp_path / "duplicate-key.parquet",
    )

    with pytest.raises(ValueError, match="duplicate metadata keys"):
        _load_and_validate_inputs([path])


def test_rejects_wavelength_and_key_mismatches_between_files(tmp_path: Path) -> None:
    """Reject mismatched wavelength and metadata-key sets across real variants."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength = next(column for column in frame.columns if column not in METADATA_COLUMNS)
    baseline = _write_variant(frame, tmp_path / "baseline.parquet")
    bad_wavelengths = _write_variant(
        frame.drop(wavelength),
        tmp_path / "bad-wavelengths.parquet",
    )
    bad_keys = _write_variant(
        frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("Time") + 1)
            .otherwise(pl.col("Time"))
            .alias("Time")
        ),
        tmp_path / "bad-keys.parquet",
    )

    with pytest.raises(ValueError, match="wavelength sets"):
        _load_and_validate_inputs([baseline, bad_wavelengths])
    with pytest.raises(ValueError, match="metadata-key sets"):
        _load_and_validate_inputs([baseline, bad_keys])


def test_t_smoothing_uses_closed_real_time_windows_on_real_fixture() -> None:
    """Average adjacent real observations without assuming equal Time spacing."""
    frame = pl.read_parquet(_fixture_paths()[0]).sort("Time")
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    times = frame["Time"].to_list()
    gaps = [right - left for left, right in pairwise(times)]
    window = min(gaps)

    without_neighbors = _apply_t_smoothing(frame, window / 2)
    with_neighbors = _apply_t_smoothing(frame, window)
    expected = [
        frame.filter(
            (pl.col("Time") >= time - window)
            & (pl.col("Time") <= time + window)
        )[wavelength].mean()
        for time in times
    ]

    assert len(set(gaps)) > 1
    assert without_neighbors.equals(frame)
    assert with_neighbors[wavelength].to_list() == pytest.approx(expected)
    assert with_neighbors.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert with_neighbors.shape == frame.shape


def test_t_smoothing_keeps_step_and_sequence_groups_separate() -> None:
    """Keep nearby rows in different Step or Sequence groups isolated."""
    frame = pl.read_parquet(_fixture_paths()[0]).head(4).with_columns(
        pl.Series("Time", [0.0, 1.0, 1.0, 2.0]),
        pl.Series("Step", [0, 0, 1, 1]),
        pl.Series("Sequence", [0, 1, 0, 0]),
    )
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )

    smoothed = _apply_t_smoothing(frame, 1.0)

    assert smoothed[wavelength][0] == pytest.approx(frame[wavelength][0])
    assert smoothed[wavelength][1] == pytest.approx(frame[wavelength][1])
    assert smoothed[wavelength][2] == pytest.approx(
        frame[wavelength].slice(2, 2).mean()
    )
    assert smoothed[wavelength][3] == pytest.approx(
        frame[wavelength].slice(2, 2).mean()
    )


def test_t_smoothing_none_preserves_real_fixture() -> None:
    """Return the validated real fixture unchanged when smoothing is disabled."""
    frame = pl.read_parquet(_fixture_paths()[0])

    assert _apply_t_smoothing(frame, None).equals(frame)


@pytest.mark.parametrize(
    "window",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_t_smoothing_rejects_invalid_windows(window: float) -> None:
    """Reject nonpositive and nonfinite t-smoothing half-window widths."""
    frame = pl.read_parquet(_fixture_paths()[0])

    with pytest.raises(ValueError, match="t_smoothing_window"):
        _apply_t_smoothing(frame, window)


def test_w_smoothing_uses_closed_real_wavelength_windows_on_real_fixture() -> None:
    """Average real wavelengths without assuming equal wavelength spacing."""
    frame = pl.read_parquet(_fixture_paths()[0]).head(3)
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    gaps = [right - left for left, right in pairwise(wavelengths)]
    window = min(gaps)

    smoothed = _apply_w_smoothing(frame, window)
    expected = [
        [
            frame.select(
                column
                for column, candidate in zip(wavelength_columns, wavelengths)
                if wavelength - window <= candidate <= wavelength + window
            ).row(row_index)
            for wavelength in wavelengths
        ]
        for row_index in range(frame.height)
    ]

    assert len(set(gaps)) > 1
    for row_index in range(frame.height):
        expected_means = [sum(values) / len(values) for values in expected[row_index]]
        assert smoothed.select(wavelength_columns).row(row_index) == pytest.approx(
            expected_means
        )
    assert smoothed.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert smoothed.shape == frame.shape


def test_w_smoothing_none_preserves_real_fixture() -> None:
    """Return the validated real fixture unchanged when smoothing is disabled."""
    frame = pl.read_parquet(_fixture_paths()[0])

    assert _apply_w_smoothing(frame, None).equals(frame)


@pytest.mark.parametrize(
    "window",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_w_smoothing_rejects_invalid_windows(window: float) -> None:
    """Reject nonpositive and nonfinite w-smoothing half-window widths."""
    frame = pl.read_parquet(_fixture_paths()[0])

    with pytest.raises(ValueError, match="w_smoothing_window"):
        _apply_w_smoothing(frame, window)


def test_t_normalization_makes_real_fixture_reference_mean_one() -> None:
    """Normalize every spectrum by its inclusive real-Time reference mean."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    times = sorted(frame["Time"].unique().to_list())
    reference_range = (times[1], times[-2])

    normalized = _apply_t_normalization(frame, reference_range)
    reference = normalized.filter(
        pl.col("Time").is_between(*reference_range, closed="both")
    )

    assert reference.select(wavelength_columns).mean().row(0) == pytest.approx(
        [1.0] * len(wavelength_columns)
    )
    assert normalized.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert normalized.shape == frame.shape


def test_t_normalization_keeps_step_and_sequence_groups_separate() -> None:
    """Use a distinct reference mean for each Step and Sequence group."""
    frame = pl.read_parquet(_fixture_paths()[0]).head(4).with_columns(
        pl.Series("Time", [0.0, 1.0, 0.0, 1.0]),
        pl.Series("Step", [0, 0, 1, 1]),
        pl.Series("Sequence", [0, 0, 1, 1]),
    )
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )

    normalized = _apply_t_normalization(frame, (0.0, 0.0))

    assert normalized[wavelength][0] == pytest.approx(1.0)
    assert normalized[wavelength][1] == pytest.approx(
        frame[wavelength][1] / frame[wavelength][0]
    )
    assert normalized[wavelength][2] == pytest.approx(1.0)
    assert normalized[wavelength][3] == pytest.approx(
        frame[wavelength][3] / frame[wavelength][2]
    )


def test_t_normalization_none_preserves_real_fixture() -> None:
    """Return the validated real fixture unchanged when normalization is disabled."""
    frame = pl.read_parquet(_fixture_paths()[0])

    assert _apply_t_normalization(frame, None).equals(frame)


@pytest.mark.parametrize(
    "reference_range",
    [
        (1.0, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0,),
        "0,1",
    ],
)
def test_t_normalization_rejects_invalid_ranges(reference_range: object) -> None:
    """Reject reversed, nonfinite, and malformed t-normalization ranges."""
    frame = pl.read_parquet(_fixture_paths()[0])

    with pytest.raises(ValueError, match="t_normalization_range"):
        _apply_t_normalization(frame, reference_range)  # type: ignore[arg-type]


def test_t_normalization_rejects_empty_or_zero_mean_reference() -> None:
    """Reject empty references and zero or nonfinite reference means."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    zero_reference = frame.with_columns(
        pl.when(pl.col("Time") == pl.col("Time").min())
        .then(0.0)
        .otherwise(pl.col(wavelength))
        .alias(wavelength)
    )
    nonfinite_reference = frame.with_columns(
        pl.when(pl.col("Time") == pl.col("Time").min())
        .then(float("inf"))
        .otherwise(pl.col(wavelength))
        .alias(wavelength)
    )

    with pytest.raises(ValueError, match="reference interval"):
        _apply_t_normalization(frame, (-2.0, -1.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_t_normalization(zero_reference, (0.0, 0.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_t_normalization(nonfinite_reference, (0.0, 0.0))


def test_w_normalization_makes_real_fixture_reference_mean_one() -> None:
    """Normalize every row by its inclusive real-wavelength reference mean."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[1], wavelengths[-2])
    reference_columns = [
        column
        for column, wavelength in zip(wavelength_columns, wavelengths)
        if reference_range[0] <= wavelength <= reference_range[1]
    ]

    normalized = _apply_w_normalization(frame, reference_range)

    assert normalized.select(reference_columns).mean_horizontal().to_list() == (
        pytest.approx([1.0] * frame.height)
    )
    assert normalized.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert normalized.shape == frame.shape


def test_w_normalization_keeps_metadata_rows_separate() -> None:
    """Use a distinct wavelength reference mean for each metadata row."""
    frame = pl.read_parquet(_fixture_paths()[0]).head(2)
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[0], wavelengths[1])
    reference_columns = wavelength_columns[:2]

    normalized = _apply_w_normalization(frame, reference_range)

    for row_index in range(frame.height):
        divisor = frame.select(reference_columns).row(row_index)
        reference_mean = sum(divisor) / len(divisor)
        assert normalized.select(wavelength_columns).row(row_index) == pytest.approx(
            [value / reference_mean for value in frame.select(wavelength_columns).row(row_index)]
        )


def test_w_normalization_none_preserves_real_fixture() -> None:
    """Return the validated real fixture unchanged when normalization is disabled."""
    frame = pl.read_parquet(_fixture_paths()[0])

    assert _apply_w_normalization(frame, None).equals(frame)


@pytest.mark.parametrize(
    "reference_range",
    [
        (1.0, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0,),
        "0,1",
    ],
)
def test_w_normalization_rejects_invalid_ranges(reference_range: object) -> None:
    """Reject reversed, nonfinite, and malformed w-normalization ranges."""
    frame = pl.read_parquet(_fixture_paths()[0])

    with pytest.raises(ValueError, match="w_normalization_range"):
        _apply_w_normalization(frame, reference_range)  # type: ignore[arg-type]


def test_w_normalization_rejects_empty_or_zero_mean_reference() -> None:
    """Reject empty references and zero or nonfinite per-row reference means."""
    frame = pl.read_parquet(_fixture_paths()[0])
    wavelength_columns = [
        column for column in frame.columns if column not in METADATA_COLUMNS
    ]
    wavelengths = [float(column.removesuffix("nm")) for column in wavelength_columns]
    reference_range = (wavelengths[0], wavelengths[1])
    zero_reference = frame.with_columns(
        pl.lit(0.0).alias(column) for column in wavelength_columns[:2]
    )
    nonfinite_reference = frame.with_columns(
        pl.lit(float("inf")).alias(column) for column in wavelength_columns[:2]
    )

    with pytest.raises(ValueError, match="reference interval"):
        _apply_w_normalization(frame, (-2.0, -1.0))
    with pytest.raises(ValueError, match="reference mean"):
        _apply_w_normalization(zero_reference, reference_range)
    with pytest.raises(ValueError, match="reference mean"):
        _apply_w_normalization(nonfinite_reference, reference_range)


def test_flatten_real_fixture_has_deterministic_columns_and_values() -> None:
    """Flatten one real Parquet fixture in numeric feature-key order."""
    path = _fixture_paths()[0]
    loaded = _load_and_validate_inputs([path])
    frame = loaded[0][1]
    wavelength_columns = sorted(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )
    metadata_rows = sorted(
        frame.select("Step", "Sequence", "Time").iter_rows(),
        key=lambda row: tuple(float(value) for value in row),
    )
    expected_columns = ["filename"] + [
        f"{wavelength}_{int(step)}_{int(sequence)}_{float(time):.2f}"
        for wavelength in wavelength_columns
        for step, sequence, time in metadata_rows
    ]
    expected_values = [
        frame.filter(
            (pl.col("Step") == step)
            & (pl.col("Sequence") == sequence)
            & (pl.col("Time") == time)
        )[wavelength].item()
        for wavelength in wavelength_columns
        for step, sequence, time in metadata_rows
    ]

    flattened = _flatten_inputs(loaded)

    assert flattened.height == 1
    assert flattened.columns == expected_columns
    assert flattened["filename"].to_list() == [path.stem]
    assert flattened.row(0)[1:] == pytest.approx(expected_values)


def test_flatten_multiple_real_fixtures_ignores_input_order() -> None:
    """Combine real fixtures in deterministic normalized-path order."""
    paths = _fixture_paths()[:3]
    loaded = _load_and_validate_inputs(paths)

    forward = _flatten_inputs(loaded)
    reverse = _flatten_inputs(list(reversed(loaded)))

    assert forward.equals(reverse)
    assert forward["filename"].to_list() == [
        path.stem for path in sorted(path.resolve() for path in paths)
    ]


def test_flatten_sorts_step_and_sequence_and_rejects_name_collisions() -> None:
    """Sort numeric metadata keys and reject colliding formatted feature names."""
    path, fixture = _load_and_validate_inputs([_fixture_paths()[0]])[0]
    frame = fixture.head(4).with_columns(
        pl.Series("Time", [1.0, 0.0, 1.0, 0.0]),
        pl.Series("Step", [1, 1, 0, 0]),
        pl.Series("Sequence", [1, 0, 1, 0]),
    )

    flattened = _flatten_inputs([(path, frame)])
    first_wavelength = min(
        (column for column in frame.columns if column not in METADATA_COLUMNS),
        key=lambda column: float(column.removesuffix("nm")),
    )

    assert flattened.columns[1:5] == [
        f"{first_wavelength}_0_0_0.00",
        f"{first_wavelength}_0_1_1.00",
        f"{first_wavelength}_1_0_0.00",
        f"{first_wavelength}_1_1_1.00",
    ]

    collision = fixture.head(2).with_columns(
        pl.Series("Time", [0.001, 0.002]),
        pl.lit(0).alias("Step"),
        pl.lit(0).alias("Sequence"),
    )
    with pytest.raises(ValueError, match="duplicate flattened feature names"):
        _flatten_inputs([(path, collision)])


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


def test_flatten_pca_runs_all_real_fixtures_end_to_end() -> None:
    """Return deterministic flattened features and PCA scores for all fixtures."""
    paths = _fixture_paths()
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


def test_readme_minimal_public_api_example_runs_on_all_real_fixtures() -> None:
    """Keep the documented optional-component real-data example executable."""
    readme = Path("README.md").read_text(encoding="utf-8")
    paths = _fixture_paths()
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
) -> None:
    """Match the specified pipeline when each preprocessing stage is enabled."""
    paths = _fixture_paths()
    expected = _expected_flatten_pca(paths, n_component=2, **preprocessing)  # type: ignore[arg-type]

    result = flatten_pca(paths, n_component=2, **preprocessing)  # type: ignore[arg-type]

    _assert_flatten_pca_matches(result, expected, 2)


def test_flatten_pca_applies_all_preprocessing_in_specified_order() -> None:
    """Apply t/w smoothing before t/w normalization using real fixture data."""
    paths = _fixture_paths()
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
def test_flatten_pca_rejects_invalid_component_counts(n_component: int) -> None:
    """Reject component counts outside one through the PCA matrix rank bound."""
    with pytest.raises(ValueError, match="n_component"):
        flatten_pca(_fixture_paths(), n_component=n_component)
