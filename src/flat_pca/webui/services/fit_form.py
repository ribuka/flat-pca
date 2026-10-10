"""Default values and validation of the preprocessing and PCA form."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, cast

import numpy as np

from flat_pca.feature_engineering.flatten_pca.preprocess_config import PreprocessConfig
from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig
from flat_pca.feature_engineering.pca.analysis import resolve_used_components

from .catalog_query import SelectionRanges

IMPUTE_STRATEGIES = ("drop", "median", "kmeans")
SCALING_STRATEGIES = ("none", "z-score", "minmax", "robust", "pareto")
INTENSITY_TRANSFORMS = ("none", "sqrt", "log1p", "asinh")
# Optional ranges entered as ``{name}_lower``/``{name}_upper`` and enabled by
# ``{name}_enabled``, with the catalog ranges supplying their initial values.
RANGE_FIELDS = ("wavelength_range", "t_normalization_range", "w_normalization_range")
WINDOW_FIELDS = ("t_smoothing_window", "w_smoothing_window")
STATISTIC_FIELDS = ("mahalanobis", "spe")
FormValues = dict[str, object]


@dataclass(frozen=True)
class FitForm:
    """A submitted form, its field errors, and the validated job settings.

    Attributes
    ----------
    values : dict[str, object]
        Submitted values for re-rendering the form: text per field, a list
        of text for ``target_steps``, and ``bool`` for checkboxes.
    errors : dict[str, str]
        Error message per field (or field group, such as
        ``wavelength_range``); empty when the form is valid.
    preprocess : dict[str, object]
        ``preprocess_and_flatten`` keyword arguments.
    pca : dict[str, object]
        ``n_component``, ``impute_strategy``, ``impute_kmeans_n_clusters``,
        and ``scaling_strategy``.
    mahalanobis : dict[str, object]
        ``MahalanobisConfig`` keyword arguments.
    spe : dict[str, object]
        ``SpeConfig`` keyword arguments.
    """

    values: FormValues
    errors: dict[str, str] = field(default_factory=dict)
    preprocess: dict[str, object] = field(default_factory=dict)
    pca: dict[str, object] = field(default_factory=dict)
    mahalanobis: dict[str, object] = field(default_factory=dict)
    spe: dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return whether the form has no errors.

        Returns
        -------
        bool
            ``True`` if ``errors`` is empty.
        """
        return not self.errors


def number_text(value: float | None) -> str:
    """Format a number for a form input.

    Parameters
    ----------
    value : float | None
        Number, or ``None``.

    Returns
    -------
    str
        ``repr``-precise text, or empty text for ``None``.
    """
    if value is None:
        return ""
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _default(config: type[Any], name: str) -> Any:
    """Return the default value of a settings dataclass field.

    Parameters
    ----------
    config : type
        Settings dataclass, e.g. ``PreprocessConfig``.
    name : str
        Field name.

    Returns
    -------
    Any
        The field's default value.

    Raises
    ------
    ValueError
        If the dataclass has no such field or the field has no default.
    """
    for config_field in fields(config):
        if config_field.name == name and config_field.default is not MISSING:
            return config_field.default
    raise ValueError(f"{config.__name__} has no default for {name}")


def default_form_values(
    ranges: SelectionRanges, overrides: Mapping[str, object] | None = None
) -> FormValues:
    """Return the initial form values.

    Every step is selected, the optional ranges are disabled with the
    catalog ranges as their initial values, and the intensity transform,
    downsampling, missing ratio, and T² and Q settings take the defaults of
    ``PreprocessConfig``, ``MahalanobisConfig``, and ``SpeConfig``. The PCA
    settings are the Web UI's own: ``n_component`` automatic, rows with
    missing values dropped, and no scaling. ``overrides`` then replaces
    any of these values.

    Parameters
    ----------
    ranges : SelectionRanges
        Steps and ranges of the selected files.
    overrides : Mapping[str, object] | None, default None
        Form values keyed by input name that replace the built-in initial
        values, such as those of ``fit_defaults.toml``.

    Returns
    -------
    dict[str, object]
        Form values keyed by input name.
    """
    values: FormValues = {
        "target_steps": [str(step) for step in ranges.steps],
        "edge_trim_start": "",
        "edge_trim_end": "",
        "wavelength_range_enabled": False,
        "wavelength_range_lower": number_text(ranges.wavelength_min),
        "wavelength_range_upper": number_text(ranges.wavelength_max),
        "t_normalization_range_enabled": False,
        "t_normalization_range_lower": number_text(ranges.time_min),
        "t_normalization_range_upper": number_text(ranges.time_max),
        "w_normalization_range_enabled": False,
        "w_normalization_range_lower": number_text(ranges.wavelength_min),
        "w_normalization_range_upper": number_text(ranges.wavelength_max),
        "t_smoothing_window": "",
        "w_smoothing_window": "",
        "intensity_transform": _default(PreprocessConfig, "intensity_transform"),
        "intensity_transform_scale": number_text(
            _default(PreprocessConfig, "intensity_transform_scale")
        ),
        "t_downsampling_stride": number_text(_default(PreprocessConfig, "t_downsampling_stride")),
        "w_downsampling_stride": number_text(_default(PreprocessConfig, "w_downsampling_stride")),
        "max_null_ratio": number_text(_default(PreprocessConfig, "max_null_ratio")),
        "n_component": "",
        "impute_strategy": "drop",
        "impute_kmeans_n_clusters": "",
        "scaling_strategy": "none",
        "mahalanobis_cumulative_explained_variance": number_text(
            _default(MahalanobisConfig, "cumulative_explained_variance")
        ),
        "mahalanobis_alpha": number_text(_default(MahalanobisConfig, "alpha")),
        "spe_cumulative_explained_variance": number_text(
            _default(SpeConfig, "cumulative_explained_variance")
        ),
        "spe_alpha": number_text(_default(SpeConfig, "alpha")),
    }
    values.update(overrides or {})
    return values


def form_values_from_config(config: Mapping[str, Any], ranges: SelectionRanges) -> FormValues:
    """Return the form values that reproduce a fit job's settings.

    Parameters
    ----------
    config : Mapping[str, Any]
        Job configuration built by ``build_fit_config``, as read by
        ``run_config``.
    ranges : SelectionRanges
        Steps and ranges of the selected files, used for disabled ranges.

    Returns
    -------
    dict[str, object]
        Form values keyed by input name.
    """
    values = default_form_values(ranges)
    preprocess = config["preprocess"]
    pca = config["pca"]
    values["target_steps"] = [str(step) for step in preprocess["target_steps"]]
    edge_trim = preprocess["edge_trim"]
    if edge_trim is not None:
        values["edge_trim_start"], values["edge_trim_end"] = map(number_text, edge_trim)
    for name in RANGE_FIELDS:
        bounds = preprocess[name]
        if bounds is not None:
            values[f"{name}_enabled"] = True
            values[f"{name}_lower"], values[f"{name}_upper"] = map(number_text, bounds)
    for name in (
        *WINDOW_FIELDS,
        "intensity_transform_scale",
        "t_downsampling_stride",
        "w_downsampling_stride",
        "max_null_ratio",
    ):
        values[name] = number_text(preprocess[name])
    values["intensity_transform"] = preprocess["intensity_transform"]
    values["n_component"] = number_text(pca["n_component"])
    values["impute_strategy"] = pca["impute_strategy"]
    values["impute_kmeans_n_clusters"] = number_text(pca["impute_kmeans_n_clusters"])
    values["scaling_strategy"] = pca["scaling_strategy"]
    for prefix in STATISTIC_FIELDS:
        statistic = config[prefix]
        for key in ("cumulative_explained_variance", "alpha"):
            values[f"{prefix}_{key}"] = number_text(statistic[key])
    return values


class _Reader:
    """Read typed values from submitted form values, recording errors.

    Parameters
    ----------
    values : Mapping[str, object]
        Submitted form values.
    errors : dict[str, str]
        Error messages, updated in place.
    """

    def __init__(self, values: Mapping[str, object], errors: dict[str, str]) -> None:
        self.values = values
        self.errors = errors

    def text(self, name: str) -> str:
        """Return a field's stripped text.

        Parameters
        ----------
        name : str
            Input name.

        Returns
        -------
        str
            The text, or empty text if absent.
        """
        value = self.values.get(name)
        return value.strip() if isinstance(value, str) else ""

    def number[T: (int, float)](
        self,
        name: str,
        convert: Callable[[str], T],
        *,
        key: str | None = None,
        optional: bool = False,
    ) -> T | None:
        """Parse a numeric field.

        Parameters
        ----------
        name : str
            Input name.
        convert : Callable[[str], int] | Callable[[str], float]
            ``int`` or ``float``.
        key : str | None, default None
            Error key; ``name`` when ``None``.
        optional : bool, default False
            Whether a blank value is allowed (and returned as ``None``).

        Returns
        -------
        int | float | None
            The parsed value, or ``None`` when blank or invalid.
        """
        raw = self.text(name)
        error_key = key or name
        if not raw:
            if not optional:
                self.errors.setdefault(error_key, "Enter a value.")
            return None
        try:
            return convert(raw)
        except ValueError:
            kind = "an integer" if convert is int else "a number"
            self.errors.setdefault(error_key, f"Enter {kind}.")
            return None

    def choice(self, name: str, choices: tuple[str, ...]) -> str:
        """Read a select field restricted to ``choices``.

        Parameters
        ----------
        name : str
            Input name.
        choices : tuple[str, ...]
            Allowed values.

        Returns
        -------
        str
            The submitted value (recorded as an error if not allowed).
        """
        value = self.text(name)
        if value not in choices:
            self.errors[name] = f"Choose one of {', '.join(choices)}."
        return value


def _check(errors: dict[str, str], key: str, validate: Callable[[], object]) -> None:
    """Record the ``ValueError`` message of a validation call.

    Parameters
    ----------
    errors : dict[str, str]
        Error messages, updated in place unless ``key`` already has one.
    key : str
        Error key.
    validate : Callable[[], object]
        Validation to run.
    """
    if key in errors:
        return
    try:
        validate()
    except ValueError as error:
        errors[key] = str(error)


def _parse_preprocess(reader: _Reader) -> dict[str, object]:
    """Parse and validate the ``PreprocessConfig`` fields.

    Each field is validated by building a ``PreprocessConfig`` with that
    field alone, so every invalid field gets its own message.

    Parameters
    ----------
    reader : _Reader
        Reader of the submitted values.

    Returns
    -------
    dict[str, object]
        ``preprocess_and_flatten`` keyword arguments (``None`` for disabled
        stages, lists for ranges).
    """
    errors = reader.errors
    preprocess: dict[str, object] = {}

    steps: list[int] = []
    for raw in cast(list[str], reader.values.get("target_steps") or []):
        try:
            steps.append(int(raw))
        except ValueError:
            errors["target_steps"] = "Steps must be integers."
    if not steps:
        errors.setdefault("target_steps", "Choose at least one Step.")
    preprocess["target_steps"] = sorted(set(steps))

    preprocess["edge_trim"] = None
    if reader.text("edge_trim_start") or reader.text("edge_trim_end"):
        bounds = [
            reader.number(f"edge_trim_{side}", float, key="edge_trim", optional=True)
            or 0.0
            for side in ("start", "end")
        ]
        preprocess["edge_trim"] = bounds
    for name in RANGE_FIELDS:
        preprocess[name] = None
        if reader.values.get(f"{name}_enabled"):
            preprocess[name] = [
                reader.number(f"{name}_{side}", float, key=name)
                for side in ("lower", "upper")
            ]
    for name in WINDOW_FIELDS:
        preprocess[name] = reader.number(name, float, optional=True)
    preprocess["intensity_transform"] = reader.choice(
        "intensity_transform", INTENSITY_TRANSFORMS
    )
    preprocess["intensity_transform_scale"] = reader.number(
        "intensity_transform_scale", float
    )
    for name in ("t_downsampling_stride", "w_downsampling_stride"):
        preprocess[name] = reader.number(name, int)
    preprocess["max_null_ratio"] = reader.number("max_null_ratio", float)

    for name in (
        "edge_trim",
        *RANGE_FIELDS,
        *WINDOW_FIELDS,
        "t_downsampling_stride",
        "w_downsampling_stride",
        "max_null_ratio",
    ):
        value = preprocess[name]
        if value is not None:
            _check(errors, name, lambda name=name, value=value: PreprocessConfig(**{name: value}))
    if "intensity_transform" not in errors:
        _check(
            errors,
            "intensity_transform_scale",
            lambda: PreprocessConfig(
                intensity_transform=preprocess["intensity_transform"],  # type: ignore[arg-type]
                intensity_transform_scale=preprocess["intensity_transform_scale"],  # type: ignore[arg-type]
            ),
        )
    return preprocess


def _parse_pca(reader: _Reader, n_files: int) -> dict[str, object]:
    """Parse and validate the PCA fields.

    Parameters
    ----------
    reader : _Reader
        Reader of the submitted values.
    n_files : int
        Number of selected files, bounding ``n_component``.

    Returns
    -------
    dict[str, object]
        ``n_component``, ``impute_strategy``, ``impute_kmeans_n_clusters``,
        and ``scaling_strategy``.
    """
    n_component = reader.number("n_component", int, optional=True)
    if n_component is not None and not 1 <= n_component <= max(n_files, 1):
        reader.errors["n_component"] = (
            f"Enter an integer from 1 to the number of selected files ({n_files})."
        )
    impute_strategy = reader.choice("impute_strategy", IMPUTE_STRATEGIES)
    n_clusters = None
    if impute_strategy == "kmeans":
        n_clusters = reader.number("impute_kmeans_n_clusters", int, optional=True)
        if n_clusters is not None and n_clusters < 1:
            reader.errors["impute_kmeans_n_clusters"] = "Enter an integer of 1 or more."
    return {
        "n_component": n_component,
        "impute_strategy": impute_strategy,
        "impute_kmeans_n_clusters": n_clusters,
        "scaling_strategy": reader.choice("scaling_strategy", SCALING_STRATEGIES),
    }


def _parse_statistic(
    reader: _Reader, prefix: str, config: type[MahalanobisConfig | SpeConfig]
) -> dict[str, object]:
    """Parse and validate the settings of the T² or Q statistic.

    Parameters
    ----------
    reader : _Reader
        Reader of the submitted values.
    prefix : str
        ``"mahalanobis"`` or ``"spe"``.
    config : type[MahalanobisConfig] | type[SpeConfig]
        Settings class validating ``alpha``.

    Returns
    -------
    dict[str, object]
        ``cumulative_explained_variance`` and ``alpha``.
    """
    errors = reader.errors
    selector_key = f"{prefix}_cumulative_explained_variance"
    alpha_key = f"{prefix}_alpha"
    selector = reader.number(selector_key, float)
    alpha = reader.number(alpha_key, float)
    if selector is not None:
        _check(
            errors,
            selector_key,
            lambda: resolve_used_components(np.ones(1), 1, selector),
        )
    if alpha is not None:
        _check(errors, alpha_key, lambda: config(alpha=alpha))
    return {"cumulative_explained_variance": selector, "alpha": alpha}


def parse_fit_form(values: Mapping[str, object], n_files: int) -> FitForm:
    """Validate the submitted preprocessing and PCA form.

    Parameters
    ----------
    values : Mapping[str, object]
        Submitted values: text per field, a list of text for
        ``target_steps``, and any value for checked checkboxes.
    n_files : int
        Number of selected files.

    Returns
    -------
    FitForm
        Field errors and, when valid, the job settings.
    """
    errors: dict[str, str] = {}
    if n_files == 0:
        errors["files"] = "Select files on Data selection."
    reader = _Reader(values, errors)
    preprocess = _parse_preprocess(reader)
    pca = _parse_pca(reader, n_files)
    mahalanobis = _parse_statistic(reader, "mahalanobis", MahalanobisConfig)
    spe = _parse_statistic(reader, "spe", SpeConfig)
    rendered: FormValues = dict(values)
    for name in RANGE_FIELDS:
        rendered[f"{name}_enabled"] = bool(values.get(f"{name}_enabled"))
    rendered["target_steps"] = list(cast(list[str], values.get("target_steps") or []))
    return FitForm(
        values=rendered,
        errors=errors,
        preprocess=preprocess,
        pca=pca,
        mahalanobis=mahalanobis,
        spe=spe,
    )
