"""Configure concise, colored console logging for command-line scripts."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{file.name}:{line}</cyan> | "
    "<level>{level.icon} {message}</level>"
)
_STANDARD_LEVELS = frozenset({"TRACE", "DEBUG", "INFO", "SUCCESS"})


def _is_standard_record(record: dict[str, Any]) -> bool:
    """Return whether a log record belongs on standard output.

    Parameters
    ----------
    record : dict[str, Any]
        Loguru record being routed to a console sink.

    Returns
    -------
    bool
        ``True`` for records below the warning level.
    """
    return record["level"].name in _STANDARD_LEVELS


def _is_error_record(record: dict[str, Any]) -> bool:
    """Return whether a log record belongs on standard error.

    Parameters
    ----------
    record : dict[str, Any]
        Loguru record being routed to a console sink.

    Returns
    -------
    bool
        ``True`` for warning, error, and critical records.
    """
    return record["level"].name not in _STANDARD_LEVELS


def configure_console_logging() -> None:
    """Configure Loguru sinks for readable command-line status messages.

    Normal status messages are written to standard output, while warnings and
    errors are written to standard error. Color is enabled independently for
    each stream only when that stream is attached to a terminal. Repeated calls
    replace existing sinks instead of duplicating output.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format=_CONSOLE_FORMAT,
        filter=_is_standard_record,
        colorize=sys.stdout.isatty(),
    )
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        filter=_is_error_record,
        colorize=sys.stderr.isatty(),
    )
