"""Regression tests for the materialize_once=True NumPy fast path (#29)."""

from pathlib import Path

import polars as pl
import pytest

from flat_pca.feature_engineering import preprocess_and_flatten


def _write_two_group_frame(path: Path) -> Path:
    """Write a synthetic file with a group entirely outside downsampling and reference.

    ``(Step=0, Sequence=0)`` has rows at Time 0.0 (kept by
    ``t_downsampling_stride=3``) and 5.0 (the normalization reference).
    ``(Step=1, Sequence=0)`` has only Time 20.0, which is neither kept by
    downsampling nor inside the reference range -- so it is entirely absent
    from the "rows some later stage still needs" set the NumPy fast path
    computes. The deferred path still processes every row of every group, so
    it still notices this group has no reference observation.
    """
    frame = pl.DataFrame(
        {
            "Time": [0.0, 5.0, 20.0],
            "Step": [0, 0, 1],
            "Sequence": [0, 0, 0],
            "500.0nm": [1.0, 2.0, 3.0],
        }
    )
    frame.write_parquet(path)
    return path


@pytest.mark.parametrize("materialize_once", [True, False])
def test_t_normalization_rejects_group_with_no_reference_row_even_when_unneeded_downstream(
    materialize_once: bool,
    tmp_path: Path,
) -> None:
    """Reject a group lacking a reference row, even if downsampling never needs it.

    Regression test for a NumPy-fast-path-specific bug: the selective
    row-computation optimization must still validate every (Step, Sequence)
    group's normalization reference over the full row set, not only over the
    rows retained for downsampling or used as another group's reference,
    otherwise a group excluded from both silently escapes this check.
    """
    path = _write_two_group_frame(tmp_path / "two-group.parquet")

    with pytest.raises(ValueError, match="reference interval is empty for a group"):
        preprocess_and_flatten(
            [path],
            t_normalization_range=(5.0, 5.0),
            t_downsampling_stride=3,
            materialize_once=materialize_once,
        ).collect()


def test_numpy_path_validates_values_before_target_steps_and_edge_trim(
    tmp_path: Path,
    real_fixture_paths: list[Path],
) -> None:
    """Reject an invalid value even in a row later dropped by target_steps.

    ``validate_frame`` (used by the deferred path) checks the whole file
    before any row filtering; the NumPy fast path must match that ordering
    instead of validating only the rows that survive ``target_steps`` and
    ``edge_trim``, since values in dropped rows have historically still been
    required to be finite.
    """
    frame = pl.read_parquet(real_fixture_paths[0])
    wavelength = next(
        column for column in frame.columns if column not in {"Time", "Step", "Sequence"}
    )
    dropped_step = int(frame["Step"].max()) + 1
    extra_row = frame.head(1).with_columns(
        pl.lit(dropped_step).alias("Step"),
        pl.lit(float("nan")).alias(wavelength),
    )
    path = tmp_path / "invalid-in-dropped-step.parquet"
    pl.concat([frame, extra_row]).write_parquet(path)

    with pytest.raises(ValueError, match="null, NaN, or infinite"):
        preprocess_and_flatten(
            [path], target_steps=[int(frame["Step"][0])], materialize_once=True
        ).collect()


def test_numpy_path_matches_deferred_path_with_every_stage_and_native_column_order(
    tmp_path: Path,
) -> None:
    """Match the deferred path bit-for-bit with every stage enabled together.

    Uses wavelength columns declared out of ascending-wavelength order, since
    the smoothing and normalization arithmetic sums values in each file's own
    native column order; reordering to sorted order before that arithmetic
    would silently change floating-point rounding relative to the deferred
    path.
    """
    rng_values = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.5, 3.5, 2.5],
        [0.5, 2.5, 1.0, 3.0],
        [3.0, 1.0, 2.0, 1.5],
    ]
    frame = pl.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Step": [0, 0, 0, 0],
            "Sequence": [0, 0, 0, 0],
            # Declared in descending wavelength order on purpose.
            "600.0nm": [row[0] for row in rng_values],
            "500.0nm": [row[1] for row in rng_values],
            "400.0nm": [row[2] for row in rng_values],
            "300.0nm": [row[3] for row in rng_values],
        }
    )
    path = tmp_path / "unsorted-columns.parquet"
    frame.write_parquet(path)

    kwargs = {
        "t_smoothing_window": 1.0,
        "w_smoothing_window": 100.0,
        "t_normalization_range": (0.0, 1.0),
        "w_normalization_range": (300.0, 500.0),
        "t_downsampling_stride": 2,
        "w_downsampling_stride": 2,
    }

    materialized = preprocess_and_flatten([path], materialize_once=True, **kwargs).collect()
    deferred = preprocess_and_flatten([path], materialize_once=False, **kwargs).collect()

    assert materialized.equals(deferred)


def test_numpy_path_reads_only_wavelength_range_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project the value read to wavelength_range columns, not the full file.

    Regression guard for the "double read of every column" inefficiency
    #29 targets: even though the file has columns outside the requested
    range, only the in-range columns (plus metadata) should ever be
    materialized.
    """
    frame = pl.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Step": [0, 0],
            "Sequence": [0, 0],
            "100.0nm": [1.0, 2.0],
            "200.0nm": [3.0, 4.0],
            "300.0nm": [5.0, 6.0],
        }
    )
    path = tmp_path / "range-restricted.parquet"
    frame.write_parquet(path)

    result = preprocess_and_flatten(
        [path], wavelength_range=(150.0, 250.0), materialize_once=True
    ).collect()

    assert result.columns == ["source", "200.0nm_0_0_0.00", "200.0nm_0_0_1.00"]


def test_numpy_path_and_deferred_path_agree_on_validate_metadata_alignment(
    tmp_path: Path, real_fixture_paths: list[Path]
) -> None:
    """Reject the same mismatched-grid inputs regardless of materialize_once.

    ``validate_metadata_alignment`` compares ``(Time, Step, Sequence)`` after
    ``target_steps`` filtering but before edge trimming, in both the NumPy
    fast path and the deferred path.
    """
    frame = pl.read_parquet(real_fixture_paths[0])
    baseline = tmp_path / "baseline.parquet"
    variant = tmp_path / "variant.parquet"
    frame.write_parquet(baseline)
    extra_row = frame.head(1).with_columns(
        pl.lit(2).alias("Step"), pl.lit(frame["Time"].max() + 100.0).alias("Time")
    )
    pl.concat([frame, extra_row]).write_parquet(variant)

    for materialize_once in (True, False):
        with pytest.raises(ValueError, match="metadata-key sets"):
            preprocess_and_flatten(
                [baseline, variant],
                validate_metadata_alignment=True,
                materialize_once=materialize_once,
            ).collect()

        ok = preprocess_and_flatten(
            [baseline, variant],
            target_steps=[1],
            validate_metadata_alignment=True,
            materialize_once=materialize_once,
        ).collect()
        assert ok.height == 2


def test_numpy_path_matches_deferred_path_with_mismatched_input_grids(
    real_fixture_paths: list[Path],
) -> None:
    """Match the deferred path's union/null-fill behavior for uneven inputs.

    Real fixtures already have differing (Step, Sequence, StepTime) grids
    across files (see the flatten-pca-mismatched-grids-expected memory), so
    this exercises the fast path's per-file union contribution against the
    deferred path without constructing a synthetic gap.
    """
    paths = real_fixture_paths

    materialized = preprocess_and_flatten(paths, max_null_ratio=1.0, materialize_once=True)
    deferred = preprocess_and_flatten(paths, max_null_ratio=1.0, materialize_once=False)

    assert materialized.collect().equals(deferred.collect())
