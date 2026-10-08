"""T² and Q settings of a shown run, read for the display screens."""

from __future__ import annotations

from collections.abc import Mapping

from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig

from ..run_config import run_config, statistic_configs
from .fit_artifacts import artifact_error


def shown_statistic_configs(run: Mapping[str, object]) -> tuple[MahalanobisConfig, SpeConfig]:
    """Return the T² and Q settings saved with a shown run.

    Parameters
    ----------
    run : Mapping[str, object]
        The fit or transform run's ``runs`` row.

    Returns
    -------
    tuple[MahalanobisConfig, SpeConfig]
        Settings saved in the run's configuration.

    Raises
    ------
    RunArtifactError
        If the saved settings are missing or invalid, so the screen asks to
        execute the run again.
    """
    try:
        return statistic_configs(run_config(run))
    except (KeyError, TypeError, ValueError) as error:
        raise artifact_error(error) from error
