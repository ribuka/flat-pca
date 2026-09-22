"""Tests for Flatten-PCA Parquet input loading and validation."""

import warnings
from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering.flatten_pca import input as _input
from flat_pca.feature_engineering.flatten_pca.input import (
    load_and_validate_inputs as _load_and_validate_inputs,
)
from flat_pca.feature_engineering.flatten_pca.input import (
    validate_frame as _validate_frame,
)
from flat_pca.feature_engineering.flatten_pca.input import (
    validate_metadata_alignment as _validate_metadata_alignment,
)
from flat_pca.spectral.schema import NON_SPECTRAL_COLUMNS

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

    assert isinstance(single[0][1], pl.LazyFrame)
    assert single[0][1].collect().equals(fixture_frame)
    assert [path for path, _ in forward] == [path for path, _ in reverse]
    assert [path for path, _ in forward] == sorted(path.resolve() for path in paths[:3])


def test_validate_frame_accepts_a_real_lazy_frame(
    real_fixture_paths: list[Path],
) -> None:
    """Validate a lazily scanned real fixture without eager input conversion."""
    path = real_fixture_paths[0]
    frame = pl.scan_parquet(path).with_columns(pl.lit(0).alias("StepTime"))

    wavelengths = _validate_frame(path, frame)

    assert wavelengths == frozenset(
        column
        for column in frame.collect_schema()
        if column not in NON_SPECTRAL_COLUMNS
    )


def test_load_passes_lazy_frames_to_validation(
    monkeypatch: pytest.MonkeyPatch,
    real_fixture_paths: list[Path],
) -> None:
    """Pass lazily scanned real fixtures to frame validation."""
    original_validate_frame = _input.validate_frame
    received_frames: list[pl.LazyFrame] = []

    def record_frame(
        path: Path,
        frame: pl.LazyFrame,
        *,
        validate_metadata_uniqueness: bool = False,
        wavelength_range: tuple[float, float] | None = None,
    ) -> frozenset[str]:
        received_frames.append(frame)
        return original_validate_frame(
            path,
            frame,
            validate_metadata_uniqueness=validate_metadata_uniqueness,
            wavelength_range=wavelength_range,
        )

    monkeypatch.setattr(_input, "validate_frame", record_frame)

    _load_and_validate_inputs(real_fixture_paths[:2])

    assert len(received_frames) == 2
    assert all(isinstance(frame, pl.LazyFrame) for frame in received_frames)


def test_rejects_empty_and_missing_paths(tmp_path: Path) -> None:
    """Reject an empty path collection and a nonexistent Parquet path."""
    with pytest.raises(ValueError, match="at least one"):
        _load_and_validate_inputs([])

    with pytest.raises(FileNotFoundError):
        _load_and_validate_inputs([tmp_path / "missing.parquet"])


def _write_duplicate_stem_variants(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> tuple[Path, Path]:
    """Write two real-fixture-derived Parquet files sharing a stem.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    real_fixture_paths : list[Path]
        Real fixture Parquet paths to derive the frame from.

    Returns
    -------
    tuple[Path, Path]
        Two written paths with identical stems in different directories.
    """
    frame = pl.read_parquet(real_fixture_paths[0])
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first = _write_variant(frame, first_directory / "sample.parquet")
    second = _write_variant(frame, second_directory / "sample.parquet")
    return first, second


def test_rejects_duplicate_path_stems_when_checked_as_error(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject duplicate stems when ``stem_uniqueness="error"`` is requested."""
    first, second = _write_duplicate_stem_variants(tmp_path, real_fixture_paths)

    with pytest.raises(ValueError, match="path stems"):
        _load_and_validate_inputs([first, second], stem_uniqueness="error")


def test_allows_duplicate_path_stems_by_default(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Load duplicate-stem inputs without error or warning by default."""
    first, second = _write_duplicate_stem_variants(tmp_path, real_fixture_paths)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = _load_and_validate_inputs([first, second])

    assert [path for path, _ in loaded] == sorted(
        (first.resolve(), second.resolve()), key=str
    )


def test_warns_on_duplicate_path_stems_when_checked_as_warn(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Warn but continue loading when ``stem_uniqueness="warn"`` is requested."""
    first, second = _write_duplicate_stem_variants(tmp_path, real_fixture_paths)

    with pytest.warns(UserWarning, match="path stems"):
        loaded = _load_and_validate_inputs([first, second], stem_uniqueness="warn")

    assert [path for path, _ in loaded] == sorted(
        (first.resolve(), second.resolve()), key=str
    )


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


def test_allows_duplicate_keys_by_default(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Allow duplicate Time, Step, and Sequence tuples by default."""
    frame = pl.read_parquet(real_fixture_paths[0])
    path = _write_variant(
        pl.concat([frame, frame.head(1)]),
        tmp_path / "duplicate-key.parquet",
    )

    loaded = _load_and_validate_inputs([path])

    assert loaded[0][0] == path.resolve()


def test_rejects_duplicate_keys_when_requested(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject duplicate Time, Step, and Sequence tuples when requested."""
    frame = pl.read_parquet(real_fixture_paths[0])
    path = _write_variant(
        pl.concat([frame, frame.head(1)]),
        tmp_path / "duplicate-key.parquet",
    )

    with pytest.raises(ValueError, match="duplicate metadata keys"):
        _load_and_validate_inputs([path], validate_metadata_uniqueness=True)


def test_rejects_wavelength_mismatches_between_files(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject mismatched wavelength sets across real variants."""
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in METADATA_COLUMNS
    )
    baseline = _write_variant(frame, tmp_path / "baseline.parquet")
    bad_wavelengths = _write_variant(
        frame.drop(wavelength),
        tmp_path / "bad-wavelengths.parquet",
    )

    with pytest.raises(ValueError, match="wavelength sets"):
        _load_and_validate_inputs([baseline, bad_wavelengths])


def test_allows_metadata_key_mismatches_between_files_by_default(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Load mismatched Time, Step, Sequence combinations without error by default."""
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = _write_variant(frame, tmp_path / "baseline.parquet")
    bad_keys = _write_variant(
        frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("Time") + 1)
            .otherwise(pl.col("Time"))
            .alias("Time")
        ),
        tmp_path / "bad-keys.parquet",
    )

    loaded = _load_and_validate_inputs([baseline, bad_keys])

    assert [path for path, _ in loaded] == sorted(
        (baseline.resolve(), bad_keys.resolve()), key=str
    )


def test_validate_metadata_alignment_accepts_matching_real_fixtures(
    real_fixture_paths: list[Path],
) -> None:
    """Accept real fixture inputs that already share Time, Step, Sequence keys."""
    loaded = _load_and_validate_inputs(real_fixture_paths)

    _validate_metadata_alignment(loaded)


def test_validate_metadata_alignment_rejects_mismatched_real_fixture_variant(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject a real-fixture-derived variant whose Time values were shifted."""
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = _write_variant(frame, tmp_path / "baseline.parquet")
    bad_keys = _write_variant(
        frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("Time") + 1)
            .otherwise(pl.col("Time"))
            .alias("Time")
        ),
        tmp_path / "bad-keys.parquet",
    )
    loaded = _load_and_validate_inputs([baseline, bad_keys])

    with pytest.raises(ValueError, match="metadata-key sets"):
        _validate_metadata_alignment(loaded)
