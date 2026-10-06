"""Validated preprocessing parameters shared by the Flatten-PCA pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..preprocess.downsampling import validate_downsampling_stride
from ..preprocess.intensity_transform import (
    IntensityTransform,
    validate_intensity_transform,
)
from ..preprocess.ranges import (
    validate_finite_bounds,
    validate_ordered_range,
    validate_positive_finite,
)
from ..preprocess.sparse_columns import validate_max_null_ratio
from ..preprocess.wavelength_filter import validate_wavelength_range
from .input import StemUniquenessCheck


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters, validated once at construction.

    Built by ``flatten_pca`` and ``preprocess_and_flatten`` from their
    keyword arguments and passed as a single object to the internal NumPy
    fast path and the polars pipeline, so every argument is validated before
    any input file is read and stages downstream receive already coerced
    values.

    Parameters
    ----------
    target_steps : list[int] | None, default None
        ``Step`` values to keep, or ``None`` to keep every row.
    edge_trim : Sequence[float] | None, default None
        Two finite StepTime/ReverseStepTime trim thresholds, or ``None`` to
        disable trimming. Stored as a ``tuple[float, float]``.
    wavelength_range : tuple[float, float] | None, default None
        Inclusive wavelength interval of columns to keep, or ``None``.
        Stored as ordered, finite ``float`` bounds.
    t_smoothing_window, w_smoothing_window : float | None, default None
        Positive finite smoothing half-windows, or ``None``. Stored as
        ``float``.
    t_normalization_range, w_normalization_range : tuple[float, float] | None, default None
        Inclusive normalization ranges, or ``None``. Stored as ordered,
        finite ``float`` bounds.
    intensity_transform : {"none", "sqrt", "log1p", "asinh"}, default "none"
        Element-wise intensity transform applied after normalization.
    intensity_transform_scale : float, default 1.0
        Positive finite divisor used by ``"log1p"`` and ``"asinh"``. Stored
        as ``float``.
    t_downsampling_stride, w_downsampling_stride : int, default 1
        Positive intervals in the shared sorted Time and wavelength arrays.
        Stored as built-in ``int``.
    max_null_ratio : float, default 0.1
        Inclusive maximum missing-value ratio for retained flattened
        feature columns.
    stem_uniqueness : Literal["skip", "warn", "error"], default "skip"
        How to handle duplicate ``Path.stem`` values across input paths.
    validate_metadata_uniqueness : bool, default False
        Whether to reject duplicate ``(Time, Step, Sequence)`` tuples within
        each input file.
    validate_metadata_alignment : bool, default False
        Whether to require every input file to share an identical set of
        ``(Time, Step, Sequence)`` tuples after ``target_steps`` filtering.

    Raises
    ------
    ValueError
        If any argument is invalid.

    See Also
    --------
    preprocess_and_flatten : Public entry point documenting each parameter.
    """

    target_steps: list[int] | None = None
    edge_trim: Sequence[float] | None = None
    wavelength_range: tuple[float, float] | None = None
    t_smoothing_window: float | None = None
    w_smoothing_window: float | None = None
    t_normalization_range: tuple[float, float] | None = None
    w_normalization_range: tuple[float, float] | None = None
    intensity_transform: IntensityTransform = "none"
    intensity_transform_scale: float = 1.0
    t_downsampling_stride: int = 1
    w_downsampling_stride: int = 1
    max_null_ratio: float = 0.1
    stem_uniqueness: StemUniquenessCheck = "skip"
    validate_metadata_uniqueness: bool = False
    validate_metadata_alignment: bool = False

    def __post_init__(self) -> None:
        """Validate every argument and store its coerced value.

        Raises
        ------
        ValueError
            If any argument is invalid.
        """
        if self.edge_trim is not None:
            self._set("edge_trim", validate_finite_bounds(self.edge_trim, "edge_trim"))
        if self.wavelength_range is not None:
            self._set("wavelength_range", validate_wavelength_range(self.wavelength_range))
        for name in ("t_smoothing_window", "w_smoothing_window"):
            window = getattr(self, name)
            if window is not None:
                self._set(name, validate_positive_finite(window, name))
        for name in ("t_normalization_range", "w_normalization_range"):
            reference_range = getattr(self, name)
            if reference_range is not None:
                self._set(name, validate_ordered_range(reference_range, name))
        transform, scale = validate_intensity_transform(
            self.intensity_transform, self.intensity_transform_scale
        )
        self._set("intensity_transform", transform)
        self._set("intensity_transform_scale", scale)
        for name in ("t_downsampling_stride", "w_downsampling_stride"):
            self._set(name, validate_downsampling_stride(getattr(self, name), name))
        validate_max_null_ratio(self.max_null_ratio)

    def _set(self, name: str, value: object) -> None:
        """Replace one field's value on this frozen instance.

        Parameters
        ----------
        name : str
            Field name.
        value : object
            Validated, coerced value to store.
        """
        object.__setattr__(self, name, value)
