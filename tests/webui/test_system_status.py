"""Tests for the memory usage and version shown in the sidebar."""

from __future__ import annotations

import re

from flat_pca.webui.services.system_status import (
    MemoryUsage,
    app_version,
    format_gib,
    read_memory_usage,
)


def test_read_memory_usage_reports_process_and_machine() -> None:
    """The process uses some memory, and the machine's use is within its total."""
    memory = read_memory_usage()

    assert memory.process_rss > 0
    assert 0 < memory.system_used <= memory.system_total
    assert 0 < memory.system_percent <= 100


def test_system_percent() -> None:
    """The share of the machine's memory is a percentage of the total."""
    memory = MemoryUsage(process_rss=1, system_used=3, system_total=12)

    assert memory.system_percent == 25


def test_format_gib() -> None:
    """Byte counts are shown in GiB with one decimal place."""
    assert format_gib(0) == "0.0 GiB"
    assert format_gib(3 * 1024**3 // 2) == "1.5 GiB"


def test_app_version_has_v_prefix() -> None:
    """The version is the package version with a ``v`` prefix."""
    assert re.fullmatch(r"v\d+\.\d+\.\d+.*", app_version())
