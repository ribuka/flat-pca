"""Time- and wavelength-direction normalization for Flatten-PCA."""

import numpy as np
import polars as pl

from flat_pca.spectral.schema import parse_wavelength, wavelength_columns
from flat_pca.utils import get_columns_from_polars

from .ranges import validate_ordered_range


def apply_t_normalization(
    frame: pl.DataFrame | pl.LazyFrame,
    t_normalization_range: tuple[float, float] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Normalize spectra by a mean from an inclusive real-Time interval.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    t_normalization_range : tuple[float, float] | None
        Inclusive reference interval in the same units as ``Time``. If
        ``None``, normalization is disabled.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Input rows and metadata with intensities divided by the reference mean
        for each Step, Sequence, and wavelength. A ``pl.LazyFrame`` input
        returns a ``pl.LazyFrame``.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; a metadata group has
        no reference observations; or a reference mean is zero or nonfinite.

    Notes
    -----
    A ``pl.LazyFrame`` input is collected exactly once and the result is
    rewrapped with ``.lazy()``, so the upstream query runs a single time.
    Validating the reference means requires materialized values, and returning
    a still-deferred query on top of that validation would re-run every
    preceding stage on the caller's own ``collect()``.
    """
    if t_normalization_range is None:
        return frame
    lower, upper = validate_ordered_range(
        t_normalization_range,
        "t_normalization_range",
    )

    if isinstance(frame, pl.LazyFrame):
        normalized = apply_t_normalization(frame.collect(), t_normalization_range)
        assert isinstance(normalized, pl.DataFrame)
        return normalized.lazy()

    spectra = wavelength_columns(frame.columns)
    source_values = frame.select(spectra).to_numpy().astype(float, copy=False)
    normalized_values = source_values.copy()
    times = frame["Time"].cast(pl.Float64).to_numpy()
    groups: dict[tuple[object, object], list[int]] = {}
    for row_index, group in enumerate(frame.select("Step", "Sequence").iter_rows()):
        groups.setdefault(group, []).append(row_index)
    if not groups:
        raise ValueError(
            "t_normalization_range reference interval is empty for a group"
        )

    for indices in groups.values():
        group_indices = np.asarray(indices)
        in_reference = (times[group_indices] >= lower) & (times[group_indices] <= upper)
        if not in_reference.any():
            raise ValueError(
                "t_normalization_range reference interval is empty for a group"
            )
        reference_mean = source_values[group_indices[in_reference]].mean(axis=0)
        if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
            raise ValueError(
                "t_normalization_range reference mean must be finite and nonzero"
            )
        normalized_values[group_indices] = source_values[group_indices] / reference_mean

    return frame.with_columns(
        pl.Series(column, normalized_values[:, column_index])
        for column_index, column in enumerate(spectra)
    )


def apply_w_normalization(
    frame: pl.DataFrame | pl.LazyFrame,
    w_normalization_range: tuple[float, float] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Normalize spectra by a mean from an inclusive wavelength interval.

    Parameters
    ----------
    frame : pl.DataFrame
        Validated Flatten-PCA input containing metadata and wavelength columns.
    w_normalization_range : tuple[float, float] | None
        Inclusive reference interval in the same units as the wavelengths. If
        ``None``, normalization is disabled.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Input rows and metadata with each row's intensities divided by its
        reference-wavelength mean. A ``pl.LazyFrame`` input returns a
        ``pl.LazyFrame``.

    Raises
    ------
    ValueError
        If the range is malformed, reversed, or nonfinite; the interval has no
        wavelengths; or a row's reference mean is zero or nonfinite.

    Notes
    -----
    A ``pl.LazyFrame`` input is collected exactly once and the result is
    rewrapped with ``.lazy()``, for the same reason as in
    ``apply_t_normalization``.
    """
    if w_normalization_range is None:
        return frame
    lower, upper = validate_ordered_range(
        w_normalization_range,
        "w_normalization_range",
    )

    columns = get_columns_from_polars(frame)
    spectra = wavelength_columns(columns)
    wavelengths = np.asarray([parse_wavelength(column) for column in spectra])
    in_reference = (wavelengths >= lower) & (wavelengths <= upper)
    if not in_reference.any():
        raise ValueError("w_normalization_range reference interval is empty")

    if isinstance(frame, pl.LazyFrame):
        normalized = apply_w_normalization(frame.collect(), w_normalization_range)
        assert isinstance(normalized, pl.DataFrame)
        return normalized.lazy()

    source_values = frame.select(spectra).to_numpy().astype(float, copy=False)
    reference_mean = source_values[:, in_reference].mean(axis=1)
    if not np.isfinite(reference_mean).all() or (reference_mean == 0).any():
        raise ValueError(
            "w_normalization_range reference mean must be finite and nonzero"
        )
    normalized_values = source_values / reference_mean[:, np.newaxis]

    return frame.with_columns(
        pl.Series(column, normalized_values[:, column_index])
        for column_index, column in enumerate(spectra)
    )
