"""Job functions run in child processes by the executor and Web UI tests.

They live in a module of their own so spawned children can import them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from flat_pca.webui.jobs.progress import write_progress

OUTPUT_FILE = "output.txt"


def write_after_delay(config: dict[str, object], run_dir: Path) -> None:
    """Sleep for ``config["seconds"]`` and then write ``output.txt``.

    Parameters
    ----------
    config : dict[str, object]
        Job configuration with ``seconds`` and ``text``.
    run_dir : Path
        Run directory.
    """
    (run_dir / "partial.txt").write_text("started", encoding="utf-8")
    time.sleep(float(config["seconds"]))
    (run_dir / OUTPUT_FILE).write_text(str(config["text"]), encoding="utf-8")


def crash(config: dict[str, object], run_dir: Path) -> None:
    """Exit the process abruptly without cleanup.

    Parameters
    ----------
    config : dict[str, object]
        Unused job configuration.
    run_dir : Path
        Unused run directory.
    """
    os._exit(3)


def raise_error(config: dict[str, object], run_dir: Path) -> None:
    """Raise an exception from the job.

    Parameters
    ----------
    config : dict[str, object]
        Unused job configuration.
    run_dir : Path
        Unused run directory.

    Raises
    ------
    ValueError
        Always.
    """
    raise ValueError("bad input")


FIT_STAGE_SECONDS = 2.0
FIT_STAGES = ("preprocess", "fit", "score", "save")
SLOW_FIT_ERROR = "slow fit stopped after its stages"


def fit_in_slow_stages(config: dict[str, object], run_dir: Path) -> None:
    """Stand in for the fit job: go through its stages slowly, then fail.

    Each stage writes ``progress.json`` like the fit job and lasts
    ``FIT_STAGE_SECONDS``, so a browser can watch the run while it is active.

    Parameters
    ----------
    config : dict[str, object]
        Unused job configuration.
    run_dir : Path
        Run directory receiving ``progress.json``.

    Raises
    ------
    RuntimeError
        Always, after the last stage.
    """
    for index, stage in enumerate(FIT_STAGES):
        write_progress(run_dir, stage, index, len(FIT_STAGES))
        time.sleep(FIT_STAGE_SECONDS)
    raise RuntimeError(SLOW_FIT_ERROR)
