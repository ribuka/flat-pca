"""Choice of the ``(Step, Sequence)`` pair shown by a heatmap."""

from __future__ import annotations

from .fit_artifacts import DisplayArtifacts


def parse_segment(text: str | None) -> tuple[int, int] | None:
    """Parse ``"{Step}:{Sequence}"``.

    Parameters
    ----------
    text : str | None
        Query parameter value.

    Returns
    -------
    tuple[int, int] | None
        The pair, or ``None`` for an empty value.

    Raises
    ------
    ValueError
        If the value is not two integers separated by ``:``.
    """
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"segment must be '<Step>:<Sequence>': {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise ValueError(f"segment must be '<Step>:<Sequence>': {text!r}") from error


def format_segment(segment: tuple[int, int] | None) -> str | None:
    """Format a pair as the ``"{Step}:{Sequence}"`` read by ``parse_segment``.

    Parameters
    ----------
    segment : tuple[int, int] | None
        The pair.

    Returns
    -------
    str | None
        Query parameter value, or ``None`` without a pair.
    """
    return None if segment is None else f"{segment[0]}:{segment[1]}"


def choose_segment(
    options: list[tuple[int, int]], requested: tuple[int, int] | None
) -> tuple[int, int] | None:
    """Return the requested segment if available, otherwise the first one.

    Parameters
    ----------
    options : list[tuple[int, int]]
        Available pairs.
    requested : tuple[int, int] | None
        Requested pair.

    Returns
    -------
    tuple[int, int] | None
        Chosen pair, or ``None`` without options.
    """
    if requested in options:
        return requested
    return options[0] if options else None


def feature_segments(artifacts: DisplayArtifacts) -> list[tuple[int, int]]:
    """Return the ``(Step, Sequence)`` pairs of a run's features.

    Parameters
    ----------
    artifacts : DisplayArtifacts
        Fit-run artifacts.

    Returns
    -------
    list[tuple[int, int]]
        Pairs in ascending order.
    """
    pairs = artifacts.features.select("Step", "Sequence").unique().sort("Step", "Sequence")
    return [(int(step), int(sequence)) for step, sequence in pairs.iter_rows()]
