"""Choice of the run a display screen shows."""

from __future__ import annotations


def choose_run(
    runs: list[dict[str, object]], requested: str | None, kind_label: str
) -> dict[str, object] | None:
    """Return the requested run, or the newest run when none is requested.

    Parameters
    ----------
    runs : list[dict[str, object]]
        Succeeded runs to choose from, newest first, as listed by
        ``list_succeeded_runs``.
    requested : str | None
        Requested run; ``None`` for the newest run.
    kind_label : str
        Run kind named in the error message, such as ``"fit"`` or
        ``"transform"``.

    Returns
    -------
    dict[str, object] | None
        The chosen run's ``runs`` row, or ``None`` when ``runs`` is empty
        and no run is requested.

    Raises
    ------
    ValueError
        If ``requested`` is not one of ``runs``.
    """
    if requested is None:
        return runs[0] if runs else None
    run = next((run for run in runs if str(run["run_id"]) == requested), None)
    if run is None:
        raise ValueError(f"succeeded {kind_label} run not found: {requested}")
    return run
