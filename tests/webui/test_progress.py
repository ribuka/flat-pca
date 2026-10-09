"""Tests for replacing ``progress.json`` while the app may be reading it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flat_pca.webui.jobs import progress as progress_module
from flat_pca.webui.jobs.progress import (
    read_progress,
    replace_progress_file,
    write_progress,
)


@pytest.fixture
def flaky_replace(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make ``os.replace`` fail with ``PermissionError`` a set number of times.

    Returns
    -------
    list[int]
        One-element list holding the remaining failures; tests set it.
    """
    remaining = [0]
    real_replace = os.replace

    def replace(source: Path, target: Path) -> None:
        if remaining[0] > 0:
            remaining[0] -= 1
            raise PermissionError(5, "Access is denied")
        real_replace(source, target)

    monkeypatch.setattr(progress_module.os, "replace", replace)
    monkeypatch.setattr(progress_module, "REPLACE_RETRY_DELAY_S", 0.0)
    return remaining


def test_write_progress_retries_while_the_file_is_in_use(
    tmp_path: Path, flaky_replace: list[int]
) -> None:
    """A transient ``PermissionError`` is retried and the progress is written."""
    flaky_replace[0] = 3

    write_progress(tmp_path, "scan", 2, 5)

    assert read_progress(tmp_path) == progress_module.Progress("scan", 2, 5)


def test_replace_gives_up_without_raising(
    tmp_path: Path, flaky_replace: list[int]
) -> None:
    """When every attempt fails the update is dropped and the old file is kept."""
    write_progress(tmp_path, "scan", 1, 5)
    source = tmp_path / "progress.json.tmp"
    source.write_text('{"stage": "scan", "done": 2, "total": 5}', encoding="utf-8")
    flaky_replace[0] = 3

    replaced = replace_progress_file(source, tmp_path / "progress.json", attempts=3)

    assert not replaced
    assert read_progress(tmp_path) == progress_module.Progress("scan", 1, 5)
