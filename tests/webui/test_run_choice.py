"""Tests for choosing the run a display screen shows."""

from __future__ import annotations

import pytest

from flat_pca.webui.services.run_choice import choose_run

RUNS: list[dict[str, object]] = [{"run_id": "new"}, {"run_id": "old"}]


def test_choose_run_defaults_to_the_newest_run() -> None:
    """Without a requested run, the first (newest) run is chosen."""
    assert choose_run(RUNS, None, "transform") is RUNS[0]


def test_choose_run_returns_the_requested_run() -> None:
    """A requested run is chosen even when it is not the newest."""
    assert choose_run(RUNS, "old", "transform") is RUNS[1]


def test_choose_run_without_runs_returns_none() -> None:
    """Without runs and without a requested run, nothing is chosen."""
    assert choose_run([], None, "fit") is None


@pytest.mark.parametrize("runs", [RUNS, []])
def test_choose_run_rejects_an_unknown_run(runs: list[dict[str, object]]) -> None:
    """A requested run that is not listed is an error naming the run kind."""
    with pytest.raises(ValueError, match="^succeeded fit run not found: gone$"):
        choose_run(runs, "gone", "fit")
