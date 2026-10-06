"""Element-wise intensity transforms that compress strong spectral peaks."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import polars as pl

from flat_pca.spectral.schema import wavelength_columns

from .ranges import validate_positive_finite

IntensityTransform = Literal["none", "sqrt", "log1p", "asinh"]

_INTENSITY_TRANSFORMS: frozenset[str] = frozenset({"none", "sqrt", "log1p", "asinh"})


def validate_intensity_transform(
    transform: str,
    scale: float,
) -> tuple[IntensityTransform, float]:
    """Validate an intensity transform name and its scale.

    Parameters
    ----------
    transform : str
        Candidate transform name.
    scale : float
        Candidate positive finite scale.

    Returns
    -------
    tuple[IntensityTransform, float]
        The validated transform name and the scale coerced to ``float``.

    Raises
    ------
    ValueError
        If ``transform`` is not one of ``"none"``, ``"sqrt"``, ``"log1p"``,
        or ``"asinh"``, or if ``scale`` is not finite and greater than 0.
    """
    if not isinstance(transform, str) or transform not in _INTENSITY_TRANSFORMS:
        raise ValueError(
            "intensity_transform must be 'none', 'sqrt', 'log1p', or 'asinh'"
        )
    validated_scale = validate_positive_finite(scale, "intensity_transform_scale")
    return cast(IntensityTransform, transform), validated_scale


def transform_intensity_values(
    values: np.ndarray,
    transform: IntensityTransform,
    scale: float,
) -> np.ndarray:
    """Apply an already validated intensity transform to a value matrix.

    Shared by ``apply_intensity_transform`` and the NumPy fast path, so both
    pipelines run the exact same float64 arithmetic.

    Parameters
    ----------
    values : np.ndarray
        Spectral intensities. Converted to ``float64``.
    transform : IntensityTransform
        Validated transform name. ``"sqrt"`` computes
        ``sign(x) * sqrt(|x|)``, ``"log1p"`` computes ``log1p(x / scale)``,
        and ``"asinh"`` computes ``asinh(x / scale)``. ``"none"`` returns
        the values unchanged.
    scale : float
        Validated positive finite scale. Not used by ``"none"`` and
        ``"sqrt"``.

    Returns
    -------
    np.ndarray
        Transformed ``float64`` values with the same shape as ``values``.

    Raises
    ------
    ValueError
        If ``transform`` is ``"log1p"`` and any ``x / scale`` is less than or
        equal to -1.
    """
    float_values = np.asarray(values, dtype=np.float64)
    if transform == "none":
        return float_values
    if transform == "sqrt":
        return np.sign(float_values) * np.sqrt(np.abs(float_values))
    scaled = float_values / scale
    if transform == "log1p":
        if (scaled <= -1.0).any():
            raise ValueError(
                "intensity_transform='log1p' requires every intensity / "
                "intensity_transform_scale to be greater than -1; "
                "use 'asinh' for data with negative values"
            )
        return np.log1p(scaled)
    return np.arcsinh(scaled)


def apply_intensity_transform(
    frame: pl.DataFrame | pl.LazyFrame,
    transform: IntensityTransform,
    scale: float = 1.0,
) -> pl.DataFrame | pl.LazyFrame:
    """Transform every spectral intensity element-wise to compress peaks.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flatten-PCA input containing metadata and wavelength columns.
    transform : {"none", "sqrt", "log1p", "asinh"}
        Element-wise transform. ``"none"`` returns ``frame`` unchanged.
        ``"sqrt"`` computes ``sign(x) * sqrt(|x|)`` so negative noise values
        are kept. ``"log1p"`` computes ``log1p(x / scale)``. ``"asinh"``
        computes ``asinh(x / scale)``, which also accepts negative values;
        ``scale`` sets the intensity from which compression starts.
    scale : float, default 1.0
        Positive finite divisor used by ``"log1p"`` and ``"asinh"``.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Input rows and metadata with transformed ``Float64`` wavelength
        columns. A ``pl.LazyFrame`` input returns a ``pl.LazyFrame``.

    Raises
    ------
    ValueError
        If ``transform`` or ``scale`` is invalid, or if ``transform`` is
        ``"log1p"`` and any ``x / scale`` is less than or equal to -1.

    Notes
    -----
    Unless ``transform`` is ``"none"``, a ``pl.LazyFrame`` input is collected
    exactly once and the result is rewrapped with ``.lazy()``, as in the
    normalization stages, because the ``"log1p"`` domain check needs
    materialized values and every transform shares the NumPy arithmetic of
    ``transform_intensity_values``.
    """
    validated_transform, validated_scale = validate_intensity_transform(transform, scale)
    if validated_transform == "none":
        return frame

    if isinstance(frame, pl.LazyFrame):
        transformed = apply_intensity_transform(
            frame.collect(), validated_transform, validated_scale
        )
        assert isinstance(transformed, pl.DataFrame)
        return transformed.lazy()

    spectra = wavelength_columns(frame.columns)
    transformed_values = transform_intensity_values(
        frame.select(spectra).to_numpy(),
        validated_transform,
        validated_scale,
    )
    return frame.with_columns(
        pl.Series(column, transformed_values[:, index], dtype=pl.Float64)
        for index, column in enumerate(spectra)
    )
