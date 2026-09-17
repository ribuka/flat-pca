"""PCA component reshaping for flattened spectral features."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from .schema import parse_wavelength

_FEATURE_NAME_PATTERN = re.compile(
    r"(?P<wavelength>[^_]+)_(?P<step>-?\d+)_(?P<sequence>-?\d+)_(?P<time>-?\d+\.\d{2})"
)


def reshape_pca_components(
    pca: PCA,
    flattened: pl.LazyFrame,
) -> np.ndarray:
    """Reshape fitted PCA components into sorted spectral coordinate axes.

    Parameters
    ----------
    pca : PCA
        Fitted PCA estimator whose components correspond to ``flattened``.
    flattened : pl.LazyFrame
        Flattened spectral features with a ``filename`` column.

    Returns
    -------
    np.ndarray
        Components with shape ``(n_components, n_wavelengths, n_steps,
        n_sequences, n_times)``. Coordinate axes are in numeric ascending
        order.

    Raises
    ------
    ValueError
        If PCA is not fitted, feature names cannot be uniquely decoded, the
        features are not a coordinate Cartesian product, or feature counts do
        not match.
    """
    feature_columns = [
        column
        for column in flattened.collect_schema().names()
        if column != "filename"
    ]
    components = _validated_components(pca, len(feature_columns))
    coordinates = [_parse_feature_coordinate(column) for column in feature_columns]
    wavelengths = sorted({coordinate[0] for coordinate in coordinates})
    steps = sorted({coordinate[1] for coordinate in coordinates})
    sequences = sorted({coordinate[2] for coordinate in coordinates})
    times = sorted({coordinate[3] for coordinate in coordinates})
    expected_count = len(wavelengths) * len(steps) * len(sequences) * len(times)
    coordinate_set = set(coordinates)
    if len(coordinate_set) != len(coordinates) or len(coordinates) != expected_count:
        raise ValueError("flattened features must form a coordinate Cartesian product")

    result = np.empty(
        (components.shape[0], len(wavelengths), len(steps), len(sequences), len(times))
    )
    wavelength_indices = {value: index for index, value in enumerate(wavelengths)}
    step_indices = {value: index for index, value in enumerate(steps)}
    sequence_indices = {value: index for index, value in enumerate(sequences)}
    time_indices = {value: index for index, value in enumerate(times)}
    for column_index, (wavelength, step, sequence, time) in enumerate(coordinates):
        result[
            :,
            wavelength_indices[wavelength],
            step_indices[step],
            sequence_indices[sequence],
            time_indices[time],
        ] = components[:, column_index]
    return result


def _validated_components(pca: PCA, feature_count: int) -> np.ndarray:
    """Return fitted PCA components after checking their feature dimension.

    Parameters
    ----------
    pca : PCA
        Candidate fitted PCA estimator.
    feature_count : int
        Number of flattened spectral feature columns.

    Returns
    -------
    np.ndarray
        Validated two-dimensional PCA component matrix.

    Raises
    ------
    ValueError
        If PCA has no compatible fitted component matrix.
    """
    try:
        components = pca.components_
        n_features = pca.n_features_in_
    except AttributeError as error:
        raise ValueError("pca must be fitted") from error
    if n_features != feature_count or components.shape[1] != feature_count:
        raise ValueError("PCA feature count does not match flattened feature count")
    return components


def _parse_feature_coordinate(column: str) -> tuple[float, int, int, float]:
    """Decode one canonical flattened feature name into numeric coordinates.

    Parameters
    ----------
    column : str
        Flattened spectral feature name.

    Returns
    -------
    tuple[float, int, int, float]
        Wavelength, Step, Sequence, and Time coordinates.

    Raises
    ------
    ValueError
        If the name is not the unique canonical flatten representation.
    """
    match = _FEATURE_NAME_PATTERN.fullmatch(column)
    if match is None:
        raise ValueError(f"cannot uniquely decode flattened feature name: {column!r}")
    wavelength_name = match["wavelength"]
    try:
        wavelength = parse_wavelength(wavelength_name)
        step = int(match["step"])
        sequence = int(match["sequence"])
        time = float(match["time"])
    except ValueError as error:
        raise ValueError(
            f"cannot uniquely decode flattened feature name: {column!r}"
        ) from error
    canonical_name = f"{wavelength_name}_{step}_{sequence}_{time:.2f}"
    if column != canonical_name:
        raise ValueError(f"cannot uniquely decode flattened feature name: {column!r}")
    return wavelength, step, sequence, time
