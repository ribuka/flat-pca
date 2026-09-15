"""Tests for Flatten-PCA Parquet input loading and validation."""

from pathlib import Path

import polars as pl
import pytest

from spca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


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


def test_loads_one_and_multiple_real_parquet_files_deterministically(
    real_fixture_paths: list[Path],
) -> None:
    """Load real fixture Parquet files in normalized path order."""
    paths = real_fixture_paths
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


def test_rejects_duplicate_path_stems(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject duplicate stems even when normalized paths differ."""
    frame = pl.read_parquet(real_fixture_paths[0])
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first = _write_variant(frame, first_directory / "sample.parquet")
    second = _write_variant(frame, second_directory / "sample.parquet")

    with pytest.raises(ValueError, match="path stems"):
        _load_and_validate_inputs([first, second])


@pytest.mark.parametrize("column", ["Time", "Step", "Sequence"])
def test_rejects_missing_required_columns(
    tmp_path: Path, column: str, real_fixture_paths: list[Path]
) -> None:
    """Reject each missing metadata column in a real-fixture variant."""
    frame = pl.read_parquet(real_fixture_paths[0]).drop(column)
    path = _write_variant(frame, tmp_path / f"missing-{column}.parquet")

    with pytest.raises(ValueError, match="required columns"):
        _load_and_validate_inputs([path])


def test_rejects_nonnumeric_metadata_and_spectra(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject nonnumeric metadata and spectral intensity columns."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
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


def test_rejects_invalid_wavelength_name(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject a noncanonical wavelength name derived from a real fixture."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
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
    real_fixture_paths: list[Path],
) -> None:
    """Reject non-finite values inserted into a real fixture."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    values = frame[wavelength].to_list()
    values[0] = invalid_value
    path = _write_variant(
        frame.with_columns(pl.Series(wavelength, values)),
        tmp_path / "invalid-value.parquet",
    )

    with pytest.raises(ValueError, match="null, NaN, or infinite"):
        _load_and_validate_inputs([path])


def test_rejects_duplicate_keys(tmp_path: Path, real_fixture_paths: list[Path]) -> None:
    """Reject duplicate Time, Step, and Sequence tuples."""
    frame = pl.read_parquet(real_fixture_paths[0])
    path = _write_variant(
        pl.concat([frame, frame.head(1)]),
        tmp_path / "duplicate-key.parquet",
    )

    with pytest.raises(ValueError, match="duplicate metadata keys"):
        _load_and_validate_inputs([path])


def test_rejects_wavelength_and_key_mismatches_between_files(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject mismatched wavelength and metadata-key sets across real variants."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
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
