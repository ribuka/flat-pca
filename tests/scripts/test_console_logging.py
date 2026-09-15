"""Tests for command-line Loguru configuration."""

import re
from collections.abc import Iterator

import pytest
from loguru import logger

from scripts.console_logging import configure_console_logging
from scripts.ralph_runner import ExitCode, main


@pytest.fixture(autouse=True)
def _reset_loguru() -> Iterator[None]:
    """Remove test sinks after each console logging test."""
    yield
    logger.remove()


def test_console_logging_routes_normal_and_error_levels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Route normal status to stdout and problems to stderr."""
    configure_console_logging()

    logger.info("task started")
    logger.success("task completed")
    logger.warning("task blocked")
    logger.error("task failed")

    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "task started" in captured.out
    assert "SUCCESS" in captured.out
    assert "task completed" in captured.out
    assert "WARNING" in captured.err
    assert "task blocked" in captured.err
    assert "ERROR" in captured.err
    assert "task failed" in captured.err
    assert "task failed" not in captured.out


def test_console_logging_includes_second_precision_and_source_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show second-precision time and the actual logging call location."""
    configure_console_logging()

    logger.info("located message")

    output = capsys.readouterr().out
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", output)
    assert re.search(r"test_console_logging\.py:\d+", output)
    assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d+", output)


def test_console_logging_disables_color_for_captured_streams(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omit ANSI escape sequences when output is not connected to a terminal."""
    configure_console_logging()

    logger.info("plain output")
    logger.error("plain error")

    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "\x1b[" not in captured.err


def test_console_logging_reconfiguration_does_not_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replace existing sinks when console logging is configured repeatedly."""
    configure_console_logging()
    configure_console_logging()

    logger.info("only once")

    assert capsys.readouterr().out.count("only once") == 1


def test_ralph_main_uses_formatted_console_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Format a representative runner error with its true source location."""
    exit_code = main(["--max-loops", "0"])

    captured = capsys.readouterr()
    assert exit_code == ExitCode.PREFLIGHT_ERROR
    assert captured.out.count("Ralph runner start") == 1
    assert captured.out.count("Ralph runner end") == 1
    assert "ERROR" in captured.err
    assert "--max-loops must be at least 1" in captured.err
    assert re.search(r"ralph_runner\.py:\d+", captured.err)
