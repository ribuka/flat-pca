"""Loading and validation of the fit form defaults in ``fit_defaults.toml``."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from .services.catalog_query import SelectionRanges
from .services.fit_form import (
    RANGE_FIELDS,
    STATISTIC_FIELDS,
    FormValues,
    default_form_values,
    number_text,
    parse_fit_form,
)
from .settings import SettingsError
from .settings_files import read_settings_document, select_settings_file

FIT_DEFAULTS_FILE_NAME = "fit_defaults.toml"
PCA_FIELDS = (
    "n_component",
    "impute_strategy",
    "impute_kmeans_n_clusters",
    "scaling_strategy",
)
# Ranges without catalog bounds: the form leaves disabled ranges blank.
_NO_RANGES = SelectionRanges(
    steps=[0],
    wavelength_min=None,
    wavelength_max=None,
    time_min=None,
    time_max=None,
    step_time_max=None,
)


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys and converted types, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RangeDefaults(_StrictModel):
    """Initial state of an optional range, e.g. ``[preprocess.wavelength_range]``.

    Attributes
    ----------
    enabled : bool
        Whether the range is enabled.
    lower, upper : float | None
        Bounds; ``None`` uses the catalog range of the selected files.
    """

    enabled: bool = False
    lower: float | None = None
    upper: float | None = None


class PreprocessDefaults(_StrictModel):
    """``[preprocess]`` section; ``None`` keeps the built-in initial value.

    Attributes
    ----------
    edge_trim_start, edge_trim_end : float | None
        Edge trim from the start and end of each ``StepTime`` range.
    wavelength_range, t_normalization_range, w_normalization_range : RangeDefaults
        Optional ranges.
    t_smoothing_window, w_smoothing_window : float | None
        Smoothing windows.
    intensity_transform : str | None
        Intensity transform.
    intensity_transform_scale : float | None
        Scale of the intensity transform.
    t_downsampling_stride, w_downsampling_stride : int | None
        Downsampling strides.
    max_null_ratio : float | None
        Maximum ratio of missing values.
    """

    edge_trim_start: float | None = None
    edge_trim_end: float | None = None
    wavelength_range: RangeDefaults = RangeDefaults()
    t_normalization_range: RangeDefaults = RangeDefaults()
    w_normalization_range: RangeDefaults = RangeDefaults()
    t_smoothing_window: float | None = None
    w_smoothing_window: float | None = None
    intensity_transform: str | None = None
    intensity_transform_scale: float | None = None
    t_downsampling_stride: int | None = None
    w_downsampling_stride: int | None = None
    max_null_ratio: float | None = None


class PcaDefaults(_StrictModel):
    """``[pca]`` section; ``None`` keeps the built-in initial value.

    Attributes
    ----------
    n_component : int | None
        Number of components; ``None`` keeps the automatic choice.
    impute_strategy : str | None
        Imputation of missing values.
    impute_kmeans_n_clusters : int | None
        Number of clusters of the ``kmeans`` imputation.
    scaling_strategy : str | None
        Scaling of the features.
    """

    n_component: PositiveInt | None = None
    impute_strategy: str | None = None
    impute_kmeans_n_clusters: PositiveInt | None = None
    scaling_strategy: str | None = None


class StatisticDefaults(_StrictModel):
    """``[mahalanobis]`` or ``[spe]`` section; ``None`` keeps the built-in value.

    Attributes
    ----------
    cumulative_explained_variance : float | None
        Component selector of the statistic.
    alpha : float | None
        Significance level of the control limit.
    """

    cumulative_explained_variance: float | None = None
    alpha: float | None = None


class FitDefaults(_StrictModel):
    """Validated contents of ``fit_defaults.toml``.

    Attributes
    ----------
    preprocess : PreprocessDefaults
        Preprocessing defaults.
    pca : PcaDefaults
        PCA defaults.
    mahalanobis : StatisticDefaults
        T² defaults.
    spe : StatisticDefaults
        Q defaults.
    """

    preprocess: PreprocessDefaults = PreprocessDefaults()
    pca: PcaDefaults = PcaDefaults()
    mahalanobis: StatisticDefaults = StatisticDefaults()
    spe: StatisticDefaults = StatisticDefaults()

    def form_values(self) -> FormValues:
        """Return the form values that replace the built-in initial values.

        Returns
        -------
        dict[str, object]
            Form values keyed by input name, holding only the values given
            in the file (and the enabled state of every range).
        """
        values: FormValues = {}
        for name, value in self.preprocess.model_dump().items():
            if name in RANGE_FIELDS:
                values[f"{name}_enabled"] = value["enabled"]
                for side in ("lower", "upper"):
                    if value[side] is not None:
                        values[f"{name}_{side}"] = number_text(value[side])
            elif value is not None:
                values[name] = _form_text(value)
        for name, value in self.pca.model_dump().items():
            if value is not None:
                values[name] = _form_text(value)
        for prefix in STATISTIC_FIELDS:
            statistic: StatisticDefaults = getattr(self, prefix)
            for name, value in statistic.model_dump().items():
                if value is not None:
                    values[f"{prefix}_{name}"] = _form_text(value)
        return values


# Defaults without a ``fit_defaults.toml``: the form's built-in initial values.
BUILTIN_FIT_DEFAULTS = FitDefaults()


def _form_text(value: str | float) -> str:
    """Format a setting value as form input text.

    Parameters
    ----------
    value : str | float
        Choice or number.

    Returns
    -------
    str
        The choice itself, or the number formatted as on the form.
    """
    return value if isinstance(value, str) else number_text(value)


def _setting_key(form_key: str) -> str:
    """Return the ``fit_defaults.toml`` key of a form field or field group.

    Parameters
    ----------
    form_key : str
        Form error key, e.g. ``"spe_alpha"`` or ``"wavelength_range"``.

    Returns
    -------
    str
        Dotted key, e.g. ``"spe.alpha"`` or ``"preprocess.wavelength_range"``.
    """
    for prefix in STATISTIC_FIELDS:
        if form_key.startswith(f"{prefix}_"):
            return f"{prefix}.{form_key.removeprefix(f'{prefix}_')}"
    if form_key in PCA_FIELDS:
        return f"pca.{form_key}"
    return f"preprocess.{form_key}"


def fit_defaults_errors(defaults: FitDefaults) -> dict[str, str]:
    """Validate the defaults with the rules of the fit form.

    A range with both bounds is checked as a pair whether or not it is
    enabled, since the bounds fill the form; a range without both bounds
    is not, since the missing bound comes from the catalog range of the
    selected files. ``n_component`` is not checked against the number of
    selected files.

    Parameters
    ----------
    defaults : FitDefaults
        Defaults to validate.

    Returns
    -------
    dict[str, str]
        Error message per dotted ``fit_defaults.toml`` key; empty when valid.
    """
    overrides = defaults.form_values()
    for name in RANGE_FIELDS:
        bounds: RangeDefaults = getattr(defaults.preprocess, name)
        overrides[f"{name}_enabled"] = (
            bounds.lower is not None and bounds.upper is not None
        )
    form = parse_fit_form(
        default_form_values(_NO_RANGES, overrides),
        n_files=defaults.pca.n_component or 1,
    )
    return {_setting_key(key): message for key, message in form.errors.items()}


def fit_defaults_path(settings_path: str | Path) -> Path:
    """Return the ``fit_defaults.toml`` path next to a settings file.

    Parameters
    ----------
    settings_path : str | Path
        Settings file path given by ``--settings``.

    Returns
    -------
    Path
        ``fit_defaults.toml`` in the settings file's directory.
    """
    return Path(settings_path).resolve().parent / FIT_DEFAULTS_FILE_NAME


def load_fit_defaults(settings_path: str | Path) -> FitDefaults:
    """Read and validate the fit form defaults next to a settings file.

    ``fit_defaults.local.toml`` is read instead of ``fit_defaults.toml``
    when it exists. Without either file, the built-in defaults are used.

    Parameters
    ----------
    settings_path : str | Path
        Settings file path given by ``--settings``.

    Returns
    -------
    FitDefaults
        Validated defaults.

    Raises
    ------
    SettingsError
        If the file cannot be read, is not valid TOML, has unknown keys or
        values of the wrong type, or has values the fit form rejects.
    """
    path = select_settings_file(fit_defaults_path(settings_path))
    if not path.is_file():
        return BUILTIN_FIT_DEFAULTS
    try:
        document = read_settings_document(path)
    except ValueError as error:
        raise SettingsError(str(error)) from error
    try:
        defaults = FitDefaults.model_validate(document)
    except ValidationError as error:
        raise SettingsError(f"invalid fit defaults in {path}:\n{error}") from error
    errors = fit_defaults_errors(defaults)
    if errors:
        details = "\n".join(f"{key}: {message}" for key, message in errors.items())
        raise SettingsError(f"invalid fit defaults in {path}:\n{details}")
    return defaults
