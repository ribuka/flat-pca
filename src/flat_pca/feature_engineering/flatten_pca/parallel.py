"""Per-file parallel execution for Flatten-PCA preprocessing.

Generic over the per-file work so both the NumPy fast path and the polars
pipeline can reuse it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor


def validate_workers(workers: object) -> int:
    """Validate a per-file worker count.

    Parameters
    ----------
    workers : object
        Requested number of worker threads.

    Returns
    -------
    int
        ``workers`` unchanged.

    Raises
    ------
    ValueError
        If ``workers`` is not an integer of at least 1. ``bool`` and
        ``None`` are rejected too; ``None`` is reserved because
        ``ThreadPoolExecutor(max_workers=None)`` means "automatic", which is
        easily mistaken for "disabled".
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1, got {workers!r}")
    return workers


def run_per_file[T, R](
    func: Callable[[T], R],
    items: Sequence[T],
    workers: int,
) -> list[R]:
    """Apply ``func`` to every item, optionally in parallel threads.

    Parameters
    ----------
    func : Callable[[T], R]
        Per-item work. Must only read shared state and write to its own
        return value.
    items : Sequence[T]
        Items to process, typically input paths.
    workers : int
        Validated worker count (see ``validate_workers``). ``1`` runs a plain
        sequential loop without creating an executor.

    Returns
    -------
    list[R]
        Results in the same order as ``items``.

    Raises
    ------
    Exception
        The exception raised for the earliest failing item in ``items``
        order, regardless of which thread failed first, so errors are
        reported deterministically. Not-yet-started items are cancelled.
    """
    if workers == 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[Future[R]] = [executor.submit(func, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
