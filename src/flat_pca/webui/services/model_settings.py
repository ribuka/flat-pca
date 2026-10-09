"""Summary of the settings a fit run's model was made with."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..run_config import run_config, statistic_configs
from .catalog_query import SelectionRanges
from .fit_form import (
    STATISTIC_FIELDS,
    FormValues,
    form_values_from_config,
    parse_fit_form,
)

# Shown for a disabled range, window, or edge trim and for an unknown count.
NONE_TEXT = "なし"
# Ranges have no catalog defaults in the summary: disabled ones are shown as NONE_TEXT.
_NO_RANGES = SelectionRanges(
    steps=[],
    wavelength_min=None,
    wavelength_max=None,
    time_min=None,
    time_max=None,
    step_time_max=None,
)
# A file count that never bounds ``n_component`` when re-validating saved settings.
_ANY_FILE_COUNT = 2**31
# The sections checked against the form; T² and Q also allow a component count.
_FORM_SECTIONS = ("preprocess", "pca")
# Values of ``jobs.artifact_dtype``.
_ARTIFACT_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class SettingsSection:
    """A titled group of setting rows.

    Attributes
    ----------
    title : str
        Group title, as the card on the preprocessing and PCA form.
    rows : tuple[tuple[str, str], ...]
        ``(label, value)`` of each setting, in the order of the form.
    """

    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ModelSettings:
    """Summary of a fit run's settings for display.

    Attributes
    ----------
    run_id : str
        The fit run.
    headline : str
        Main values shown while the summary is folded; empty with an error.
    sections : tuple[SettingsSection, ...]
        Every setting grouped as on the form; empty with an error.
    error : str | None
        Why the saved settings cannot be read, or ``None``.
    """

    run_id: str
    headline: str = ""
    sections: tuple[SettingsSection, ...] = ()
    error: str | None = None


def _or_none(text: object) -> str:
    """Return a value's text, or ``NONE_TEXT`` when it is blank or missing.

    Parameters
    ----------
    text : object
        Form value text, a count, or ``None``.

    Returns
    -------
    str
        The text, or ``NONE_TEXT``.
    """
    return NONE_TEXT if text is None or text == "" else str(text)


def _pair(values: FormValues, lower: str, upper: str, enabled: bool) -> str:
    """Format a pair of bounds as ``lower – upper``.

    Parameters
    ----------
    values : FormValues
        Form values of the run.
    lower, upper : str
        Input names of the bounds.
    enabled : bool
        Whether the pair is set.

    Returns
    -------
    str
        ``lower – upper``, or ``NONE_TEXT`` when not set.
    """
    return f"{values[lower]} – {values[upper]}" if enabled else NONE_TEXT


def _range(values: FormValues, name: str, label: str) -> tuple[str, str]:
    """Return the row of an optional range.

    Parameters
    ----------
    values : FormValues
        Form values of the run.
    name : str
        Range field, e.g. ``wavelength_range``.
    label : str
        Label of the row.

    Returns
    -------
    tuple[str, str]
        ``(label, "lower – upper")``, or ``NONE_TEXT`` as the value when disabled.
    """
    enabled = bool(values[f"{name}_enabled"])
    return label, _pair(values, f"{name}_lower", f"{name}_upper", enabled)


def _validated_values(config: Mapping[str, Any]) -> FormValues:
    """Return the form values of saved settings, checking their types and values.

    The preprocessing and PCA settings are validated by the form itself
    (``parse_fit_form``) and must equal the saved ones, so that values of a
    wrong type, such as a string of Steps, do not pass as plausible text.
    The T² and Q settings, whose ``cumulative_explained_variance`` may also
    be a component count unlike on the form, must be numbers accepted by
    ``statistic_configs``.

    Parameters
    ----------
    config : Mapping[str, Any]
        A fit run's configuration (see ``build_fit_config``).

    Returns
    -------
    FormValues
        Form values of the settings (see ``form_values_from_config``).

    Raises
    ------
    KeyError, TypeError, ValueError
        If the configuration lacks a setting or holds an invalid value.
    """
    values = form_values_from_config(config, _NO_RANGES)
    form = parse_fit_form(values, n_files=_ANY_FILE_COUNT)
    errors = [
        (name, message)
        for name, message in form.errors.items()
        if not name.startswith(STATISTIC_FIELDS)
    ]
    if errors:
        name, message = errors[0]
        raise ValueError(f"invalid {name}: {message}")
    for name in _FORM_SECTIONS:
        if getattr(form, name) != config[name]:
            raise ValueError(f"invalid {name}: {config[name]!r}")
    statistic_configs(config)
    for prefix in STATISTIC_FIELDS:
        for key, value in config[prefix].items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"invalid {prefix}_{key}: {value!r}")
    if config["artifact_dtype"] not in _ARTIFACT_DTYPES:
        raise ValueError(f"invalid artifact_dtype: {config['artifact_dtype']!r}")
    return values


def _sections(
    run: Mapping[str, object], config: Mapping[str, Any]
) -> tuple[SettingsSection, ...]:
    """Build the setting rows of a fit run.

    Parameters
    ----------
    run : Mapping[str, object]
        The fit run's ``runs`` row, giving its counts.
    config : Mapping[str, Any]
        The run's configuration (see ``build_fit_config``).

    Returns
    -------
    tuple[SettingsSection, ...]
        Targets, Target Steps, Preprocessing, PCA, T² / Q, and Artifacts.

    Raises
    ------
    KeyError, TypeError, ValueError
        If the configuration lacks a setting or holds an invalid value.
    """
    values = _validated_values(config)
    edge_trim = values["edge_trim_start"] != "" or values["edge_trim_end"] != ""
    kmeans = values["impute_strategy"] == "kmeans"
    return (
        SettingsSection(
            "Targets",
            (
                ("Files", _or_none(run.get("n_files"))),
                ("Features", _or_none(run.get("n_features"))),
                ("Fitted components", _or_none(run.get("n_components"))),
            ),
        ),
        SettingsSection(
            "Target Steps",
            (("Target Steps", ", ".join(values["target_steps"]) or NONE_TEXT),),  # type: ignore[arg-type]
        ),
        SettingsSection(
            "Preprocessing",
            (
                _range(values, "wavelength_range", "Wavelength range"),
                (
                    "Edge trim (StepTime)",
                    _pair(values, "edge_trim_start", "edge_trim_end", edge_trim),
                ),
                (
                    "Time smoothing (half width, Time)",
                    _or_none(values["t_smoothing_window"]),
                ),
                (
                    "Wavelength smoothing (half width, nm)",
                    _or_none(values["w_smoothing_window"]),
                ),
                _range(
                    values, "t_normalization_range", "Time normalization range (Time)"
                ),
                _range(
                    values,
                    "w_normalization_range",
                    "Wavelength normalization range (nm)",
                ),
                ("Intensity transform", str(values["intensity_transform"])),
                ("Transform scale", str(values["intensity_transform_scale"])),
                ("Time downsampling stride", str(values["t_downsampling_stride"])),
                (
                    "Wavelength downsampling stride",
                    str(values["w_downsampling_stride"]),
                ),
                ("max_null_ratio", str(values["max_null_ratio"])),
            ),
        ),
        SettingsSection(
            "PCA",
            (
                ("Components", str(values["n_component"]) or "auto"),
                ("Imputation", str(values["impute_strategy"])),
                (
                    "kmeans clusters",
                    (str(values["impute_kmeans_n_clusters"]) or "default")
                    if kmeans
                    else NONE_TEXT,
                ),
                ("Scaling", str(values["scaling_strategy"])),
            ),
        ),
        SettingsSection(
            "T² / Q",
            (
                (
                    "T² cumulative explained variance",
                    str(values["mahalanobis_cumulative_explained_variance"]),
                ),
                ("T² α", str(values["mahalanobis_alpha"])),
                (
                    "Q cumulative explained variance",
                    str(values["spe_cumulative_explained_variance"]),
                ),
                ("Q α", str(values["spe_alpha"])),
            ),
        ),
        SettingsSection(
            "Artifacts", (("Artifact dtype", str(config["artifact_dtype"])),)
        ),
    )


def model_settings(run: Mapping[str, object]) -> ModelSettings:
    """Summarize the settings a fit run was made with.

    Every setting is shown whether set or not: a disabled range, window,
    or edge trim is ``NONE_TEXT``, an automatic component count ``auto``,
    and the kmeans cluster count ``NONE_TEXT`` unless imputing with kmeans.

    Parameters
    ----------
    run : Mapping[str, object]
        A fit run's ``runs`` row.

    Returns
    -------
    ModelSettings
        The summary, or an error if the saved configuration lacks a setting
        or holds an invalid value.
    """
    run_id = str(run["run_id"])
    try:
        config = run_config(run)
        sections = _sections(run, config)
        pca = config["pca"]
        components = run.get("n_components")
        if components is None:
            components = pca["n_component"] or "auto"
        headline = f"{components} components · scaling {pca['scaling_strategy']}"
    except (KeyError, TypeError, ValueError) as error:
        detail = f"missing {error}" if isinstance(error, KeyError) else str(error)
        return ModelSettings(
            run_id=run_id, error=f"Cannot read the settings of {run_id}: {detail}"
        )
    return ModelSettings(run_id=run_id, headline=headline, sections=sections)
