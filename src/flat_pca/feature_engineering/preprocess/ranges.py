"""Interval-argument validation shared by the preprocessing stages."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def validate_finite_bounds(
    value: Sequence[float],
    argument_name: str,
) -> tuple[float, float]:
    """Validate and coerce a pair of finite numeric bounds.

    The two bounds are validated independently, so this function places no
    ordering constraint on them. Use it for arguments whose two values are
    unrelated thresholds rather than the ends of one interval.

    Parameters
    ----------
    value : Sequence[float]
        Candidate two-element ``tuple`` or ``list`` of bounds.
    argument_name : str
        Public argument name used in validation messages.

    Returns
    -------
    tuple[float, float]
        Finite floating-point bounds in their original order.

    Raises
    ------
    ValueError
        If the candidate is malformed, non-numeric, boolean, or nonfinite.
    """
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(bound, bool) for bound in value)
    ):
        raise ValueError(f"{argument_name} must contain two finite bounds")
    try:
        lower, upper = (float(bound) for bound in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{argument_name} must contain two finite bounds") from error
    if not isfinite(lower) or not isfinite(upper):
        raise ValueError(f"{argument_name} must contain two finite bounds")
    return lower, upper


def validate_ordered_range(
    value: Sequence[float],
    argument_name: str,
) -> tuple[float, float]:
    """Validate and coerce an inclusive ``(lower, upper)`` interval.

    Parameters
    ----------
    value : Sequence[float]
        Candidate two-element ``tuple`` or ``list`` of bounds.
    argument_name : str
        Public argument name used in validation messages.

    Returns
    -------
    tuple[float, float]
        Finite, ordered floating-point bounds.

    Raises
    ------
    ValueError
        If the candidate is malformed, nonfinite, or reversed.
    """
    lower, upper = validate_finite_bounds(value, argument_name)
    if lower > upper:
        raise ValueError(f"{argument_name} must contain ordered finite bounds")
    return lower, upper
