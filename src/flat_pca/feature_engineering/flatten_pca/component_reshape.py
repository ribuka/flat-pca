"""PCA component reshaping for flattened spectral features."""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from flat_pca.spectral.schema import flattened_feature_columns, parse_wavelength

from ..pca import PcaModel

_FEATURE_NAME_PATTERN = re.compile(
    r"(?P<wavelength>[^_]+)_(?P<step>-?\d+)_(?P<sequence>-?\d+)_(?P<time>-?\d+\.\d{2})"
)


def reshape_pca_components(
    pca_model: PcaModel,
    flattened: pl.LazyFrame,
) -> pl.DataFrame:
    """Reshape fitted PCA components into sorted spectral coordinate axes.

    Parameters
    ----------
    pca_model : PcaModel
        Fitted PCA pipeline state whose components correspond to ``flattened``.
    flattened : pl.LazyFrame
        Flattened spectral features with a ``source`` column.

    Returns
    -------
    pl.DataFrame
        Long-form components ordered by Step, Sequence, StepTime, component,
        and wavelength.

    Raises
    ------
    ValueError
        If PCA is not fitted, feature names cannot be uniquely decoded, the
        features are not a coordinate Cartesian product, or feature counts do
        not match.
    """
    feature_columns = flattened_feature_columns(flattened.collect_schema().names())
    components = _validated_components(pca_model, len(feature_columns))
    coordinates = [_parse_feature_coordinate(column) for column in feature_columns]
    axes, order = _sorted_feature_order(coordinates)
    steps, sequences, times, wavelengths = axes
    n_components = components.shape[0]
    block_count = len(steps) * len(sequences) * len(times)

    # Rows follow (Step, Sequence, StepTime, component, wavelength). Because the
    # features form the full Cartesian product of the four coordinate axes, every
    # row position is known in advance and no sort is needed.
    coefficients = (
        components[:, order]
        .reshape(n_components, block_count, len(wavelengths))
        .transpose(1, 0, 2)
        .ravel()
    )
    component_axis = np.arange(1, n_components + 1, dtype=np.int64)
    step_grid, sequence_grid, time_grid, component_grid, wavelength_grid = np.meshgrid(
        steps, sequences, times, component_axis, wavelengths, indexing="ij", copy=False
    )
    return pl.DataFrame(
        {
            "StepTime": time_grid.ravel(),
            "Step": step_grid.ravel(),
            "Sequence": sequence_grid.ravel(),
            "wavelength": wavelength_grid.ravel(),
            "component": component_grid.ravel(),
            "coefficient": coefficients,
        }
    )


def _sorted_feature_order(
    coordinates: list[tuple[float, int, int, float]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return sorted coordinate axes and the feature order that follows them.

    Parameters
    ----------
    coordinates : list[tuple[float, int, int, float]]
        Wavelength, Step, Sequence, and StepTime coordinates of each feature.

    Returns
    -------
    tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Ascending Step, Sequence, StepTime, and wavelength axes, and the
        feature indices ordered by Step, Sequence, StepTime, and wavelength.

    Raises
    ------
    ValueError
        If the features are not the full Cartesian product of the axes.
    """
    wavelengths, wavelength_index = np.unique(
        np.array([coordinate[0] for coordinate in coordinates], dtype=np.float64),
        return_inverse=True,
    )
    steps, step_index = np.unique(
        np.array([coordinate[1] for coordinate in coordinates], dtype=np.int64),
        return_inverse=True,
    )
    sequences, sequence_index = np.unique(
        np.array([coordinate[2] for coordinate in coordinates], dtype=np.int64),
        return_inverse=True,
    )
    times, time_index = np.unique(
        np.array([coordinate[3] for coordinate in coordinates], dtype=np.float64),
        return_inverse=True,
    )
    shape = (len(steps), len(sequences), len(times), len(wavelengths))
    feature_count = len(coordinates)
    if feature_count != int(np.prod(shape)):
        raise ValueError("flattened features must form a coordinate Cartesian product")
    positions = np.ravel_multi_index(
        (step_index, sequence_index, time_index, wavelength_index), shape
    )
    order = np.full(feature_count, -1, dtype=np.intp)
    order[positions] = np.arange(feature_count, dtype=np.intp)
    if (order < 0).any():
        raise ValueError("flattened features must form a coordinate Cartesian product")
    return (steps, sequences, times, wavelengths), order


def _validated_components(pca_model: PcaModel, feature_count: int) -> np.ndarray:
    """Return fitted PCA components after checking their feature dimension.

    Parameters
    ----------
    pca_model : PcaModel
        Candidate fitted PCA pipeline state.
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
    if not hasattr(pca_model.pca, "components_") or not hasattr(
        pca_model.pca, "n_features_in_"
    ):
        raise ValueError("pca must be fitted")
    components = pca_model.pca.components_
    n_features = pca_model.pca.n_features_in_
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
        Wavelength, Step, Sequence, and StepTime coordinates.

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
