"""Tests for the element-wise intensity transform preprocessing stage."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering import flatten_pca, preprocess_and_flatten
from flat_pca.feature_engineering.preprocess import apply_intensity_transform

TRANSFORMS = ["sqrt", "log1p", "asinh"]


def _spectral_frame() -> pl.DataFrame:
    """Return a small frame with a strong peak column and a negative value.

    Returns
    -------
    pl.DataFrame
        ``Time``, ``Step``, ``Sequence`` and two wavelength columns.
    """
    return pl.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Step": [0, 0, 0],
            "Sequence": [0, 0, 0],
            "500.0nm": [1000.0, 4000.0, 9000.0],
            "600.0nm": [-0.5, 0.0, 2.0],
        }
    )


def _write_input_files(tmp_path: Path) -> list[Path]:
    """Write two positive-valued files on a shared grid with a peak column.

    Wavelength columns are declared in descending order so the arithmetic
    runs in each file's native column order, as in real inputs.

    Parameters
    ----------
    tmp_path : Path
        Directory to write into.

    Returns
    -------
    list[Path]
        Paths of the written Parquet files.
    """
    rng = np.random.default_rng(0)
    paths: list[Path] = []
    for file_index in range(2):
        n_rows = 12
        values = rng.uniform(0.5, 2.0, size=(n_rows, 4))
        values[:, 1] *= 1000.0
        frame = pl.DataFrame(
            {
                "Time": [float(index % 6) for index in range(n_rows)],
                "Step": [index // 6 for index in range(n_rows)],
                "Sequence": [0] * n_rows,
                "700.0nm": values[:, 0],
                "600.0nm": values[:, 1],
                "500.0nm": values[:, 2],
                "400.0nm": values[:, 3],
            }
        )
        path = tmp_path / f"input-{file_index}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    ("transform", "scale", "expected"),
    [
        ("sqrt", 1.0, lambda x: np.sign(x) * np.sqrt(np.abs(x))),
        ("sqrt", 5.0, lambda x: np.sign(x) * np.sqrt(np.abs(x))),
        ("log1p", 2.0, lambda x: np.log1p(x / 2.0)),
        ("asinh", 0.5, lambda x: np.arcsinh(x / 0.5)),
    ],
)
def test_apply_intensity_transform_matches_formula(
    transform: str,
    scale: float,
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Transform wavelength columns only, by the documented formula."""
    frame = _spectral_frame()

    result = apply_intensity_transform(frame, transform, scale)  # type: ignore[arg-type]

    assert isinstance(result, pl.DataFrame)
    assert result.columns == frame.columns
    assert result.select("Time", "Step", "Sequence").equals(
        frame.select("Time", "Step", "Sequence")
    )
    assert result.schema["500.0nm"] == pl.Float64
    np.testing.assert_array_equal(
        result.select("500.0nm", "600.0nm").to_numpy(),
        expected(frame.select("500.0nm", "600.0nm").to_numpy()),
    )


def test_apply_intensity_transform_none_returns_input_unchanged() -> None:
    """Return the input itself when the transform is disabled."""
    frame = _spectral_frame()

    assert apply_intensity_transform(frame, "none") is frame


def test_apply_intensity_transform_keeps_lazy_frames_lazy() -> None:
    """Return a LazyFrame with the eager result for a LazyFrame input."""
    frame = _spectral_frame()

    result = apply_intensity_transform(frame.lazy(), "asinh", 2.0)

    assert isinstance(result, pl.LazyFrame)
    expected = apply_intensity_transform(frame, "asinh", 2.0)
    assert isinstance(expected, pl.DataFrame)
    assert result.collect().equals(expected)


def test_apply_intensity_transform_compresses_strong_peak() -> None:
    """Shrink the ratio between a strong peak and a weak value."""
    frame = _spectral_frame()

    result = apply_intensity_transform(frame, "log1p")

    assert isinstance(result, pl.DataFrame)
    raw_ratio = frame["500.0nm"][2] / frame["600.0nm"][2]
    transformed_ratio = result["500.0nm"][2] / result["600.0nm"][2]
    assert transformed_ratio < raw_ratio / 100.0


@pytest.mark.parametrize(("value", "scale"), [(-2.0, 2.0), (-1.0, 1.0), (-5.0, 1.0)])
def test_apply_intensity_transform_rejects_log1p_domain_error(
    value: float, scale: float
) -> None:
    """Reject ``log1p`` when any ``x / scale`` is less than or equal to -1."""
    frame = _spectral_frame().with_columns(pl.lit(value).alias("600.0nm"))

    with pytest.raises(ValueError, match="greater than -1"):
        apply_intensity_transform(frame, "log1p", scale)


def test_apply_intensity_transform_accepts_log1p_values_above_minus_one() -> None:
    """Accept negative values whose ``x / scale`` stays above -1."""
    frame = _spectral_frame().with_columns(pl.lit(-1.5).alias("600.0nm"))

    result = apply_intensity_transform(frame, "log1p", 2.0)

    assert isinstance(result, pl.DataFrame)
    np.testing.assert_allclose(result["600.0nm"].to_numpy(), np.log1p(-0.75))


@pytest.mark.parametrize(
    ("transform", "scale", "match"),
    [
        ("log", 1.0, "intensity_transform must be"),
        (None, 1.0, "intensity_transform must be"),
        ("asinh", 0.0, "intensity_transform_scale"),
        ("asinh", -1.0, "intensity_transform_scale"),
        ("asinh", float("inf"), "intensity_transform_scale"),
        ("asinh", float("nan"), "intensity_transform_scale"),
        ("asinh", True, "intensity_transform_scale"),
        ("none", "abc", "intensity_transform_scale"),
    ],
)
def test_apply_intensity_transform_rejects_invalid_arguments(
    transform: object, scale: object, match: str
) -> None:
    """Reject an unknown transform or a non-positive, nonfinite scale."""
    with pytest.raises(ValueError, match=match):
        apply_intensity_transform(_spectral_frame(), transform, scale)  # type: ignore[arg-type]


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_numpy_path_matches_deferred_path_with_intensity_transform(
    transform: str, tmp_path: Path
) -> None:
    """Match the polars path bit-for-bit with every stage enabled."""
    paths = _write_input_files(tmp_path)
    kwargs = {
        "t_smoothing_window": 1.0,
        "w_smoothing_window": 100.0,
        "t_normalization_range": (0.0, 1.0),
        "w_normalization_range": (400.0, 500.0),
        "intensity_transform": transform,
        "intensity_transform_scale": 0.5,
        "t_downsampling_stride": 2,
        "w_downsampling_stride": 2,
    }

    materialized = preprocess_and_flatten(paths, materialize_once=True, **kwargs).collect()
    deferred = preprocess_and_flatten(paths, materialize_once=False, **kwargs).collect()

    assert materialized.equals(deferred)


@pytest.mark.parametrize("transform", ["sqrt", "asinh"])
def test_numpy_path_matches_deferred_path_with_negative_intensities(
    transform: str, tmp_path: Path
) -> None:
    """Match the polars path bit-for-bit for sign-preserving transforms."""
    path = tmp_path / "negative.parquet"
    _spectral_frame().write_parquet(path)

    materialized = preprocess_and_flatten(
        [path], intensity_transform=transform, materialize_once=True
    ).collect()
    deferred = preprocess_and_flatten(
        [path], intensity_transform=transform, materialize_once=False
    ).collect()

    assert materialized.equals(deferred)
    assert (materialized.drop("source").to_numpy() < 0).any()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_numpy_path_matches_deferred_path_on_real_fixtures(
    transform: str, real_fixture_paths: list[Path]
) -> None:
    """Match the polars path bit-for-bit on real spectra."""
    kwargs = {"intensity_transform": transform, "w_downsampling_stride": 2}

    materialized = preprocess_and_flatten(
        real_fixture_paths, materialize_once=True, **kwargs
    ).collect()
    deferred = preprocess_and_flatten(
        real_fixture_paths, materialize_once=False, **kwargs
    ).collect()

    assert materialized.equals(deferred)


@pytest.mark.parametrize("materialize_once", [True, False])
def test_intensity_transform_is_applied_after_normalization(
    materialize_once: bool, tmp_path: Path
) -> None:
    """Transform the normalized intensities, so the scale is in normalized units."""
    paths = _write_input_files(tmp_path)
    kwargs = {
        "t_normalization_range": (0.0, 1.0),
        "materialize_once": materialize_once,
    }

    normalized = preprocess_and_flatten(paths, **kwargs).collect()
    transformed = preprocess_and_flatten(
        paths, intensity_transform="asinh", intensity_transform_scale=0.5, **kwargs
    ).collect()

    assert transformed.columns == normalized.columns
    assert transformed["source"].equals(normalized["source"])
    np.testing.assert_allclose(
        transformed.drop("source").to_numpy(),
        np.arcsinh(normalized.drop("source").to_numpy() / 0.5),
        rtol=1e-15,
    )


@pytest.mark.parametrize("materialize_once", [True, False])
def test_log1p_domain_is_checked_only_on_values_kept_by_downsampling(
    materialize_once: bool, tmp_path: Path
) -> None:
    """Ignore a log1p domain error in a dropped row and reject one in a kept row.

    Both pipelines check the domain on exactly the values that survive
    downsampling, so they agree on whether the call raises.
    """
    frame = pl.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Step": [0, 0, 0],
            "Sequence": [0, 0, 0],
            "500.0nm": [1.0, -3.0, 2.0],
        }
    )
    path = tmp_path / "domain.parquet"
    frame.write_parquet(path)

    kept = preprocess_and_flatten(
        [path],
        intensity_transform="log1p",
        t_downsampling_stride=2,
        materialize_once=materialize_once,
    ).collect()
    assert kept.width == 3

    with pytest.raises(ValueError, match="greater than -1"):
        preprocess_and_flatten(
            [path], intensity_transform="log1p", materialize_once=materialize_once
        ).collect()


@pytest.mark.parametrize("materialize_once", [True, False])
def test_flatten_pca_forwards_intensity_transform(
    materialize_once: bool, real_fixture_paths: list[Path]
) -> None:
    """Fit the same PCA as on features preprocessed with the transform."""
    kwargs = {
        "intensity_transform": "asinh",
        "intensity_transform_scale": 0.1,
        "materialize_once": materialize_once,
    }

    model = flatten_pca(real_fixture_paths, n_component=2, **kwargs)
    expected = flatten_pca(
        flattened=preprocess_and_flatten(real_fixture_paths, **kwargs), n_component=2
    )

    np.testing.assert_array_equal(model.pca.components_, expected.pca.components_)
    np.testing.assert_array_equal(model.pca.mean_, expected.pca.mean_)
