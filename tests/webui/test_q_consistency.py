"""Tests for comparing Q contribution totals with the saved Q."""

from __future__ import annotations

import pytest

from flat_pca.webui.services.q_consistency import QMismatch, q_mismatch


def test_totals_within_the_tolerance_agree() -> None:
    """A relative difference up to 1% is not reported."""
    assert q_mismatch("s-00", 1.0099, 1.0) is None
    assert q_mismatch("s-00", 0.0, 0.0) is None


def test_totals_beyond_the_tolerance_are_reported() -> None:
    """A larger difference is reported with both values."""
    mismatch = q_mismatch("s-01", 0.05369221, 0.90889259)

    assert mismatch == QMismatch(stem="s-01", contribution_total=0.05369221, saved_q=0.90889259)
    assert mismatch.relative_difference == pytest.approx(
        (0.90889259 - 0.05369221) / 0.90889259
    )


def test_relative_difference_of_a_zero_saved_q_is_infinite() -> None:
    """A nonzero total against a zero Q is infinitely far off."""
    mismatch = q_mismatch("s-02", 1e-3, 0.0)

    assert mismatch is not None
    assert mismatch.relative_difference == float("inf")
