"""Tests for the aligned per-row container of the NumPy fast path (#56)."""

import numpy as np
import pytest

from flat_pca.feature_engineering.flatten_pca.file_rows import FileRows


@pytest.fixture
def file_rows() -> FileRows:
    """Return three aligned rows whose fields encode their row position."""
    return FileRows(
        times=np.array([0.0, 1.0, 2.0]),
        steps=[10, 11, 12],
        sequences=[20, 21, 22],
        step_times=np.array([0.5, 1.5, 2.5]),
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_take_reduces_every_field_together(file_rows: FileRows) -> None:
    """Keep the same rows, in the requested order, across every field."""
    taken = file_rows.take(np.array([2, 0]))

    np.testing.assert_array_equal(taken.times, [2.0, 0.0])
    assert taken.steps == [12, 10]
    assert taken.sequences == [22, 20]
    np.testing.assert_array_equal(taken.step_times, [2.5, 0.5])
    np.testing.assert_array_equal(taken.values, [[5.0, 6.0], [1.0, 2.0]])


def test_take_uses_values_computed_for_the_kept_rows(file_rows: FileRows) -> None:
    """Use the supplied values instead of reducing the original matrix."""
    computed = np.array([[7.0], [8.0]])

    taken = file_rows.take(np.array([0, 2]), values=computed)

    assert taken.values is computed
    assert taken.steps == [10, 12]


def test_take_rejects_values_with_a_different_row_count(file_rows: FileRows) -> None:
    """Reject supplied values that do not have one row per kept position."""
    with pytest.raises(ValueError, match="same row count"):
        file_rows.take(np.array([0, 2]), values=np.zeros((3, 2)))


def test_rejects_misaligned_fields() -> None:
    """Reject construction when any field has a different row count."""
    with pytest.raises(ValueError, match="same row count"):
        FileRows(
            times=np.array([0.0, 1.0]),
            steps=[10],
            sequences=[20, 21],
            step_times=np.array([0.5, 1.5]),
            values=np.zeros((2, 1)),
        )
