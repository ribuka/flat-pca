"""PCA component reshaping for flattened spectral features."""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from ..pca import PcaModel
from .schema import parse_wavelength

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
        Long-form components ordered by Step, Sequence, Time, component, and
        wavelength.

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
        if column != "source"
    ]
    components = _validated_components(pca_model, len(feature_columns))
    coordinates = [_parse_feature_coordinate(column) for column in feature_columns]
    wavelengths = sorted({coordinate[0] for coordinate in coordinates})
    steps = sorted({coordinate[1] for coordinate in coordinates})
    sequences = sorted({coordinate[2] for coordinate in coordinates})
    times = sorted({coordinate[3] for coordinate in coordinates})
    expected_count = len(wavelengths) * len(steps) * len(sequences) * len(times)
    coordinate_set = set(coordinates)
    if len(coordinate_set) != len(coordinates) or len(coordinates) != expected_count:
        raise ValueError("flattened features must form a coordinate Cartesian product")

    component_columns = [str(index) for index in range(components.shape[0])]
    table = pl.DataFrame(
        {
            "wavelength": [coordinate[0] for coordinate in coordinates],
            "Step": [coordinate[1] for coordinate in coordinates],
            "Sequence": [coordinate[2] for coordinate in coordinates],
            "Time": [coordinate[3] for coordinate in coordinates],
        }
    ).with_columns(
        [
            pl.Series(name, components[index, :])
            for index, name in enumerate(component_columns)
        ]
    )
    return (
        table.unpivot(
            index=["wavelength", "Step", "Sequence", "Time"],
            on=component_columns,
            variable_name="component",
            value_name="coefficient",
        )
        .with_columns(pl.col("component").cast(pl.Int64))
        .select(["Time", "Step", "Sequence", "wavelength", "component", "coefficient"])
        .sort(["Step", "Sequence", "Time", "component", "wavelength"])
    )


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
