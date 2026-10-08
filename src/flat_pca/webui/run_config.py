"""Reading the configuration saved with a fit or transform run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig


def run_config(run: Mapping[str, object]) -> dict[str, Any]:
    """Return the configuration a run was executed with.

    Parameters
    ----------
    run : Mapping[str, object]
        A fit or transform run's ``runs`` row.

    Returns
    -------
    dict[str, Any]
        The run's ``config_json``, as built by ``build_fit_config`` or
        ``build_transform_config``.
    """
    return json.loads(str(run["config_json"]))


def statistic_configs(config: Mapping[str, Any]) -> tuple[MahalanobisConfig, SpeConfig]:
    """Return the T² and Q settings of a run configuration.

    Parameters
    ----------
    config : Mapping[str, Any]
        A fit or transform run's configuration, holding ``mahalanobis``
        and ``spe`` keyword arguments.

    Returns
    -------
    tuple[MahalanobisConfig, SpeConfig]
        The T² and Q settings.

    Raises
    ------
    KeyError
        If the configuration lacks either setting.
    TypeError
        If a setting holds an unknown keyword.
    ValueError
        If a setting holds an invalid value.
    """
    return MahalanobisConfig(**config["mahalanobis"]), SpeConfig(**config["spe"])
