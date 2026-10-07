"""Choice of a 1-based component number among a run's components."""

from __future__ import annotations


def choose_component(requested: int | None, default: int, component_count: int) -> int:
    """Return the requested component number if available, otherwise ``default``.

    Parameters
    ----------
    requested : int | None
        Requested 1-based component number.
    default : int
        Component number used without a valid request; it is capped at
        ``component_count``.
    component_count : int
        Number of components of the run.

    Returns
    -------
    int
        Chosen 1-based component number.
    """
    if requested is not None and 1 <= requested <= component_count:
        return requested
    return min(default, component_count)
