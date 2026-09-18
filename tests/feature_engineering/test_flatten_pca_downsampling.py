"""Tests for Flatten-PCA downsampling stages."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
)
from spca.feature_engineering.flatten_pca.downsampling import (
    apply_t_downsampling as _apply_t_downsampling,
)
from spca.feature_engineering.flatten_pca.downsampling import (
    apply_w_downsampling as _apply_w_downsampling,
)
from spca.feature_engineering.flatten_pca.downsampling import (
    collect_unique_times as _collect_unique_times,
)
from spca.feature_engineering.flatten_pca.downsampling import (
    collect_unique_wavelengths as _collect_unique_wavelengths,
)
from spca.feature_engineering.flatten_pca.flatten import flatten_inputs
from spca.feature_engineering.flatten_pca.input import load_and_validate_inputs
from spca.feature_engineering.flatten_pca.normalization import (
    apply_t_normalization,
    apply_w_normalization,
)
from spca.feature_engineering.flatten_pca.smoothing import (
    apply_t_smoothing,
    apply_w_smoothing,
)

_METADATA_COLUMNS = ["Time", "Step", "Sequence"]


def test_t_downsampling_collects_sorted_unique_times_from_all_real_inputs(
    real_fixture_paths: list[Path],
) -> None:
    """Collect one numeric ascending Unique array from every real fixture."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]

    unique_times = _collect_unique_times(frames)

    expected = sorted(
        {
            float(time)
            for frame in frames
            for time in frame.get_column("Time").to_list()
        }
    )
    assert unique_times == expected
    assert all(isinstance(time, float) for time in unique_times)


@pytest.mark.parametrize("stride", [1, 2])
def test_t_downsampling_keeps_selected_real_rows_for_every_group(
    stride: int,
    real_fixture_paths: list[Path],
) -> None:
    """Keep all real Step and Sequence rows at selected Time indices."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]
    unique_times = _collect_unique_times(frames)
    frame = frames[0]
    selected_times = unique_times[::stride]

    downsampled = _apply_t_downsampling(frame, unique_times, stride)
    expected = frame.filter(pl.col("Time").is_in(selected_times))

    assert downsampled.equals(expected)
    assert downsampled.get_column("Time").unique().sort().to_list() == selected_times
    assert downsampled.select("Time", "Step", "Sequence").equals(
        expected.select("Time", "Step", "Sequence")
    )


def test_t_downsampling_does_not_add_unselected_tail(
    real_fixture_paths: list[Path],
) -> None:
    """Drop a real-data-derived final Time when its index is not selected."""
    frame = pl.read_parquet(real_fixture_paths[0]).head(6)
    unique_times = _collect_unique_times([frame])

    downsampled = _apply_t_downsampling(frame, unique_times, 2)

    assert downsampled.get_column("Time").unique().sort().to_list() == unique_times[::2]
    assert unique_times[-1] not in downsampled.get_column("Time").to_list()


@pytest.mark.parametrize("stride", [0, -1, True, False, 1.5, "2", None])
def test_t_downsampling_rejects_invalid_stride(
    stride: object,
    real_fixture_paths: list[Path],
) -> None:
    """Reject zero, negative, boolean, and non-integer intervals."""
    frame = pl.read_parquet(real_fixture_paths[0])
    unique_times = _collect_unique_times([frame])

    with pytest.raises(ValueError, match="t_downsampling_stride"):
        _apply_t_downsampling(frame, unique_times, stride)  # type: ignore[arg-type]


def test_w_downsampling_collects_sorted_unique_wavelengths_from_all_real_inputs(
    real_fixture_paths: list[Path],
) -> None:
    """Collect one numeric ascending wavelength array from every real fixture."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]

    unique_wavelengths = _collect_unique_wavelengths(frames)

    expected = sorted(
        {
            float(column.removesuffix("nm"))
            for frame in frames
            for column in frame.columns
            if column not in _METADATA_COLUMNS
        }
    )
    assert unique_wavelengths == expected
    assert all(isinstance(wavelength, float) for wavelength in unique_wavelengths)


@pytest.mark.parametrize("stride", [1, 2])
def test_w_downsampling_keeps_selected_real_columns_and_metadata(
    stride: int,
    real_fixture_paths: list[Path],
) -> None:
    """Keep real wavelength columns at selected indices and all metadata."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]
    unique_wavelengths = _collect_unique_wavelengths(frames)
    frame = frames[0]
    selected_wavelengths = unique_wavelengths[::stride]
    expected_columns = [
        *_METADATA_COLUMNS,
        *(f"{wavelength:.1f}nm" for wavelength in selected_wavelengths),
    ]

    downsampled = _apply_w_downsampling(frame, unique_wavelengths, stride)

    assert downsampled.columns == expected_columns
    assert downsampled.equals(frame.select(expected_columns))


def test_w_downsampling_does_not_add_unselected_tail(
    real_fixture_paths: list[Path],
) -> None:
    """Drop a real-data-derived final wavelength when its index is not selected."""
    source = pl.read_parquet(real_fixture_paths[0])
    wavelength_columns = [
        column for column in source.columns if column not in _METADATA_COLUMNS
    ]
    frame = source.select(*_METADATA_COLUMNS, *wavelength_columns[:6])
    unique_wavelengths = _collect_unique_wavelengths([frame])

    downsampled = _apply_w_downsampling(frame, unique_wavelengths, 2)

    expected_wavelengths = unique_wavelengths[::2]
    assert downsampled.columns == [
        *_METADATA_COLUMNS,
        *(f"{wavelength:.1f}nm" for wavelength in expected_wavelengths),
    ]
    assert f"{unique_wavelengths[-1]:.1f}nm" not in downsampled.columns


@pytest.mark.parametrize("stride", [0, -1, True, False, 1.5, "2", None])
def test_w_downsampling_rejects_invalid_stride(
    stride: object,
    real_fixture_paths: list[Path],
) -> None:
    """Reject zero, negative, boolean, and non-integer intervals."""
    frame = pl.read_parquet(real_fixture_paths[0])
    unique_wavelengths = _collect_unique_wavelengths([frame])

    with pytest.raises(ValueError, match="w_downsampling_stride"):
        _apply_w_downsampling(  # type: ignore[arg-type]
            frame,
            unique_wavelengths,
            stride,
        )


def _expected_downsampled_flatten(
    paths: list[Path],
    *,
    t_stride: int,
    w_stride: int,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Build expected flattened features in the required stage order.

    Parameters
    ----------
    paths : list[Path]
        Real Parquet fixture paths.
    t_stride : int
        Time downsampling interval.
    w_stride : int
        Wavelength downsampling interval.
    t_smoothing_window : float | None, default None
        Time smoothing half-window.
    w_smoothing_window : float | None, default None
        Wavelength smoothing half-window.
    t_normalization_range : tuple[float, float] | None, default None
        Inclusive Time normalization interval.
    w_normalization_range : tuple[float, float] | None, default None
        Inclusive wavelength normalization interval.

    Returns
    -------
    pl.DataFrame
        Flattened features after all preprocessing and downsampling stages.
    """
    loaded = load_and_validate_inputs(paths)
    frames = [frame for _, frame in loaded]
    unique_times = _collect_unique_times(frames)
    unique_wavelengths = _collect_unique_wavelengths(frames)
    prepared = []
    for path, frame in loaded:
        transformed = apply_t_smoothing(frame, t_smoothing_window)
        transformed = apply_w_smoothing(transformed, w_smoothing_window)
        transformed = apply_t_normalization(
            transformed,
            t_normalization_range,
        )
        transformed = apply_w_normalization(
            transformed,
            w_normalization_range,
        )
        transformed = _apply_t_downsampling(
            transformed,
            unique_times,
            t_stride,
        )
        transformed = _apply_w_downsampling(
            transformed,
            unique_wavelengths,
            w_stride,
        )
        prepared.append((path, transformed))
    flattened = flatten_inputs(prepared)
    assert isinstance(flattened, pl.LazyFrame)
    return flattened.collect()


@pytest.mark.parametrize(
    ("t_stride", "w_stride"),
    [(2, 1), (1, 2), (2, 2)],
)
def test_flatten_pca_downsamples_real_inputs_end_to_end(
    t_stride: int,
    w_stride: int,
    real_fixture_paths: list[Path],
) -> None:
    """Keep the shared downsampled feature set for every real input."""
    expected = _expected_downsampled_flatten(
        real_fixture_paths,
        t_stride=t_stride,
        w_stride=w_stride,
    )

    flattened = preprocess_and_flatten(
        list(reversed(real_fixture_paths)),
        t_downsampling_stride=t_stride,
        w_downsampling_stride=w_stride,
    )
    result = append_pca_scores(
        flatten_pca(
            list(reversed(real_fixture_paths)),
            n_component=2,
            t_downsampling_stride=t_stride,
            w_downsampling_stride=w_stride,
        ),
        flattened,
    ).collect()

    feature_columns = result.columns[1:-2]
    assert feature_columns == expected.columns[1:]
    assert len(feature_columns) == expected.width - 1
    assert result.get_column("filename").to_list() == expected["filename"].to_list()
    assert result.select(feature_columns).equals(expected.select(feature_columns))
    assert result.columns[-2:] == ["pca-1", "pca-2"]
    assert result.select(pl.exclude("filename")).to_numpy().shape == (
        len(real_fixture_paths),
        len(feature_columns) + 2,
    )
    assert result.select(pl.exclude("filename")).to_numpy().dtype.kind == "f"
    assert np.isfinite(result.select(pl.exclude("filename")).to_numpy()).all()


def test_flatten_pca_preprocesses_all_observations_before_downsampling(
    real_fixture_paths: list[Path],
) -> None:
    """Apply smoothing and normalization before either downsampling stage."""
    frames = [pl.read_parquet(path) for path in real_fixture_paths]
    times = _collect_unique_times(frames)
    wavelengths = _collect_unique_wavelengths(frames)
    preprocessing = {
        "t_smoothing_window": times[1] - times[0],
        "w_smoothing_window": wavelengths[1] - wavelengths[0],
        "t_normalization_range": (times[1], times[1]),
        "w_normalization_range": (wavelengths[1], wavelengths[1]),
    }
    expected = _expected_downsampled_flatten(
        real_fixture_paths,
        t_stride=2,
        w_stride=2,
        **preprocessing,
    )

    flattened = preprocess_and_flatten(
        real_fixture_paths,
        t_downsampling_stride=2,
        w_downsampling_stride=2,
        **preprocessing,
    )
    result = append_pca_scores(
        flatten_pca(
            real_fixture_paths,
            n_component=2,
            t_downsampling_stride=2,
            w_downsampling_stride=2,
            **preprocessing,
        ),
        flattened,
    ).collect()

    assert result.columns[1:-2] == expected.columns[1:]
    assert result.select(result.columns[1:-2]).to_numpy() == pytest.approx(
        expected.select(expected.columns[1:]).to_numpy()
    )


def test_flatten_pca_stride_one_preserves_existing_end_to_end_result(
    real_fixture_paths: list[Path],
) -> None:
    """Preserve legacy flattened and PCA results for default stride one."""
    default = flatten_pca(real_fixture_paths, n_component=3)
    explicit = flatten_pca(
        real_fixture_paths,
        n_component=3,
        t_downsampling_stride=1,
        w_downsampling_stride=1,
    )

    assert explicit.pca.n_components_ == default.pca.n_components_ == 3
    assert explicit.pca.mean_ == pytest.approx(default.pca.mean_)
    assert explicit.pca.components_ == pytest.approx(default.pca.components_)


def test_flatten_pca_uses_downsampled_feature_count_for_pca_limit(
    real_fixture_paths: list[Path],
    tmp_path: Path,
) -> None:
    """Bound PCA components by features remaining after downsampling."""
    derived_paths = []
    for path in real_fixture_paths:
        derived_path = tmp_path / path.name
        pl.read_parquet(path).head(1).write_parquet(derived_path)
        derived_paths.append(derived_path)

    result = flatten_pca(
        derived_paths,
        t_downsampling_stride=99,
        w_downsampling_stride=99,
    )

    assert result.pca.n_components_ == 1
    with pytest.raises(ValueError, match="n_component"):
        flatten_pca(
            derived_paths,
            n_component=2,
            t_downsampling_stride=99,
            w_downsampling_stride=99,
        )


@pytest.mark.parametrize("stride", [0, -1, True, 1.5, "2", None])
@pytest.mark.parametrize(
    "argument_name",
    ["t_downsampling_stride", "w_downsampling_stride"],
)
def test_flatten_pca_rejects_invalid_downsampling_stride(
    argument_name: str,
    stride: object,
    real_fixture_paths: list[Path],
) -> None:
    """Reject invalid downsampling intervals through the public API."""
    with pytest.raises(ValueError, match=argument_name):
        flatten_pca(  # type: ignore[arg-type]
            real_fixture_paths,
            n_component=1,
            **{argument_name: stride},
        )
