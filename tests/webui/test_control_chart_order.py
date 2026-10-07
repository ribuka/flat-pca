"""Tests for the file order of the T² and Q control charts."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from flat_pca.webui.services.monitoring import control_chart_order


def _samples() -> pl.DataFrame:
    """Return five files with a date, a lot, and a missing date."""
    return pl.DataFrame(
        {
            "stem": ["run-10", "run-2", "run-1", "run-3", "run-20"],
            "date": [
                date(2024, 1, 3),
                None,
                date(2024, 1, 1),
                date(2024, 1, 3),
                date(2024, 1, 2),
            ],
            "lot": ["B", "A", "B", None, "A"],
        }
    )


def _stems(samples: pl.DataFrame, order_by: str | None) -> list[str]:
    """Return the stems in control-chart order."""
    return samples["stem"].gather(control_chart_order(samples, order_by)).to_list()


def test_natural_order_without_a_column() -> None:
    """Without a column, the stems are in natural order."""
    assert _stems(_samples(), None) == ["run-1", "run-2", "run-3", "run-10", "run-20"]


def test_date_column_orders_ascending_with_missing_last() -> None:
    """Dates ascend, ties fall back to the stems, and missing dates come last."""
    assert _stems(_samples(), "date") == ["run-1", "run-20", "run-3", "run-10", "run-2"]


def test_category_column_breaks_ties_in_natural_order() -> None:
    """Equal categories keep the natural order of their stems."""
    assert _stems(_samples(), "lot") == ["run-2", "run-20", "run-1", "run-10", "run-3"]


def test_empty_samples() -> None:
    """An empty frame has an empty order."""
    assert _stems(_samples().clear(), "date") == []


@pytest.mark.parametrize("name", ["row", "stem_rank", "order_value", "index", "__stem_rank"])
def test_metadata_column_may_share_an_internal_name(name: str) -> None:
    """A column named like an internal sort column still orders the rows."""
    samples = pl.DataFrame({"stem": ["s-10", "s-2", "s-1"], name: [30, 10, 10]})

    np.testing.assert_array_equal(control_chart_order(samples, name), [2, 1, 0])
