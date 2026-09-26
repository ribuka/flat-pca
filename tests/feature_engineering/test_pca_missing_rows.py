"""Tests for NumPy-side removal of rows with missing feature values."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flat_pca.feature_engineering.pca import fit as fit_module
from flat_pca.feature_engineering.pca import fit_pca, transform_pca
from flat_pca.feature_engineering.pca.missing_rows import (
    collect_complete_rows,
    complete_row_mask,
    select_rows,
)

N_ROWS = 12
N_COLUMNS = 40
COLUMNS = [f"f{index}" for index in range(N_COLUMNS)]


def _frame(missing: dict[tuple[int, int], float | None]) -> pl.DataFrame:
    """Build a random feature frame with a ``source`` column and given gaps.

    Parameters
    ----------
    missing : dict[tuple[int, int], float | None]
        ``(row, column)`` positions mapped to the missing marker to place
        there: ``None`` for null or ``float("nan")`` for NaN.

    Returns
    -------
    pl.DataFrame
        Frame with ``source`` and ``f0`` onward feature columns.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(N_ROWS, N_COLUMNS)).tolist()
    for (row, column), marker in missing.items():
        values[row][column] = marker
    return pl.DataFrame(
        {
            "source": [f"s{row}" for row in range(N_ROWS)],
            **{
                name: [values[row][index] for row in range(N_ROWS)]
                for index, name in enumerate(COLUMNS)
            },
        },
        schema={"source": pl.String, **{name: pl.Float64 for name in COLUMNS}},
    )


MISSING_CASES = {
    "no-missing": {},
    "null": {(1, 3): None, (7, 39): None},
    "nan": {(0, 0): float("nan"), (5, 20): float("nan")},
    "null-and-nan": {(2, 10): None, (2, 11): float("nan"), (9, 30): float("nan")},
}


def _legacy_drop(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop rows the way the per-column Polars filter did."""
    return frame.filter(
        ~pl.any_horizontal(
            [pl.col(column).is_null() | pl.col(column).is_nan() for column in COLUMNS]
        )
    )


class TestCompleteRowMask:
    """Tests for ``complete_row_mask``."""

    @pytest.mark.parametrize("chunk_columns", [1, 7, 40, 1000])
    def test_matches_unchunked_mask(self, chunk_columns: int) -> None:
        """Give the same mask regardless of the chunk width."""
        values = np.ones((4, 40))
        values[1, 0] = np.nan
        values[3, 39] = np.nan

        mask = complete_row_mask(values, chunk_columns=chunk_columns)

        assert mask.tolist() == [True, False, True, False]

    def test_accepts_integer_matrix(self) -> None:
        """Treat every row of an integer matrix as complete."""
        assert complete_row_mask(np.ones((3, 2), dtype=np.int64)).all()

    def test_rejects_non_positive_chunk(self) -> None:
        """Reject a chunk width that would never advance."""
        with pytest.raises(ValueError, match="chunk_columns"):
            complete_row_mask(np.ones((1, 1)), chunk_columns=0)


class TestCollectCompleteRows:
    """Tests for ``collect_complete_rows`` and ``select_rows``."""

    @pytest.mark.parametrize("missing", MISSING_CASES.values(), ids=MISSING_CASES)
    def test_matches_polars_filter_bit_for_bit(
        self, missing: dict[tuple[int, int], float | None]
    ) -> None:
        """Keep exactly the rows and values the Polars filter kept."""
        frame = _frame(missing)
        expected = _legacy_drop(frame)

        rows, mask = collect_complete_rows(frame.lazy(), COLUMNS)

        assert np.array_equal(rows, expected.select(COLUMNS).to_numpy())
        assert_frame_equal(select_rows(frame.lazy(), mask).collect(), expected)

    def test_returns_empty_matrix_when_every_row_is_missing(self) -> None:
        """Return zero rows when each row has a missing value."""
        frame = _frame({(row, row): None for row in range(N_ROWS)})

        rows, mask = collect_complete_rows(frame.lazy(), COLUMNS)

        assert rows.shape == (0, N_COLUMNS)
        assert not mask.any()
        assert select_rows(frame.lazy(), mask).collect().height == 0


class TestNumpyDropPath:
    """Compare the NumPy drop path of ``fit_pca``/``transform_pca`` to Polars."""

    @staticmethod
    def _fit_and_transform(frame: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        """Fit and transform with the configuration ``fit_flattened_pca`` uses."""
        model = fit_pca(
            frame.lazy(),
            COLUMNS,
            n_component=3,
            max_n_component=None,
            impute_strategy="drop",
            outlier_strategy=None,
            scaling_strategy="none",
        )
        return model.pca.components_, transform_pca(frame.lazy(), model).collect()

    @pytest.mark.parametrize("missing", MISSING_CASES.values(), ids=MISSING_CASES)
    def test_matches_polars_path_bit_for_bit(
        self,
        missing: dict[tuple[int, int], float | None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fit the same components and return the same scored rows."""
        frame = _frame(missing)
        components, transformed = self._fit_and_transform(frame)

        monkeypatch.setattr(
            fit_module, "_drops_missing_rows_in_numpy", lambda *_: False
        )
        legacy_components, legacy_transformed = self._fit_and_transform(frame)

        assert np.array_equal(components, legacy_components)
        assert_frame_equal(transformed, legacy_transformed)
        assert transformed["source"].to_list() == (
            _legacy_drop(frame)["source"].to_list()
        )

    def test_rejects_when_no_rows_remain(self) -> None:
        """Keep the empty-result error of the Polars path."""
        frame = _frame({(row, 0): float("nan") for row in range(N_ROWS)})

        with pytest.raises(ValueError, match="no rows remain after preprocessing"):
            self._fit_and_transform(frame)

    def test_transform_rejects_when_no_rows_remain(self) -> None:
        """Raise from ``transform_pca`` when every input row is dropped."""
        model = fit_pca(
            _frame({}).lazy(),
            COLUMNS,
            n_component=2,
            impute_strategy="drop",
            scaling_strategy="none",
        )
        frame = _frame({(row, 1): None for row in range(N_ROWS)})

        with pytest.raises(ValueError, match="no rows remain after preprocessing"):
            transform_pca(frame.lazy(), model)
