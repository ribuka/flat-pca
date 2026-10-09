"""Tests for aggregating PCA loadings by a feature key."""

import numpy as np
import polars as pl
import pytest

from flat_pca.feature_engineering.flatten_pca import (
    aggregate_loadings,
    reshape_pca_components,
)
from flat_pca.feature_engineering.pca import fit_pca


@pytest.fixture
def components() -> pl.DataFrame:
    """Return long-form coefficients of two components over two wavelengths.

    Returns
    -------
    pl.DataFrame
        Component 1 has coefficients ``3, -1`` at 400 nm and ``2, 2`` at
        410 nm; component 2 has ``-4, 0`` and ``1, -1``.
    """
    return pl.DataFrame(
        {
            "StepTime": [0.0, 1.0] * 4,
            "Step": [1] * 8,
            "Sequence": [1] * 8,
            "wavelength": [400.0, 400.0, 410.0, 410.0] * 2,
            "component": [1, 1, 1, 1, 2, 2, 2, 2],
            "coefficient": [3.0, -1.0, 2.0, 2.0, -4.0, 0.0, 1.0, -1.0],
        }
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", [1.0, 2.0, -2.0, 0.0]),
        ("rms", [np.sqrt(5.0), 2.0, np.sqrt(8.0), 1.0]),
        ("abs_mean", [2.0, 2.0, 2.0, 1.0]),
    ],
)
def test_aggregates_each_component_and_wavelength(
    components: pl.DataFrame, method: str, expected: list[float]
) -> None:
    """Each method reduces the coefficients of one wavelength to one value."""
    result = aggregate_loadings(components, method)  # type: ignore[arg-type]

    assert result.columns == ["component", "wavelength", "loading"]
    assert result["component"].to_list() == [1, 1, 2, 2]
    assert result["wavelength"].to_list() == [400.0, 410.0, 400.0, 410.0]
    np.testing.assert_allclose(result["loading"].to_numpy(), expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", [2.5, 0.5, -1.5, -0.5]),
        ("rms", [np.sqrt(6.5), np.sqrt(2.5), np.sqrt(8.5), np.sqrt(0.5)]),
        ("abs_mean", [2.5, 1.5, 2.5, 0.5]),
    ],
)
def test_aggregates_by_the_given_column(
    components: pl.DataFrame, method: str, expected: list[float]
) -> None:
    """Grouping by ``StepTime`` reduces the coefficients over the wavelengths."""
    result = aggregate_loadings(components, method, by="StepTime")  # type: ignore[arg-type]

    assert result.columns == ["component", "StepTime", "loading"]
    assert result["component"].to_list() == [1, 1, 2, 2]
    assert result["StepTime"].to_list() == [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(result["loading"].to_numpy(), expected)


def test_accepts_reshape_pca_components_output() -> None:
    """The reshaped components aggregate to one row per component and wavelength."""
    rng = np.random.default_rng(5)
    columns = [
        f"{wavelength}_{step}_1_{time:.2f}"
        for wavelength in ("400.0nm", "410.0nm", "420.0nm")
        for step in (1, 2)
        for time in (0.0, 1.0)
    ]
    frame = pl.DataFrame(rng.normal(size=(10, len(columns))), schema=columns).with_columns(
        pl.lit("file").alias("source")
    )
    model = fit_pca(frame.lazy(), columns, max_n_component=2)

    reshaped = reshape_pca_components(model, frame.lazy())
    result = aggregate_loadings(reshaped, "abs_mean")

    assert result.height == 2 * 3
    first = np.abs(model.pca.components_[0][[columns.index(c) for c in columns[:4]]]).mean()
    assert result.row(0, named=True)["loading"] == pytest.approx(first)


def test_rejects_unknown_methods(components: pl.DataFrame) -> None:
    """Only the three documented methods are accepted."""
    with pytest.raises(ValueError, match="method must be one of"):
        aggregate_loadings(components, "median")  # type: ignore[arg-type]


def test_rejects_frames_without_coefficients(components: pl.DataFrame) -> None:
    """The coefficient, component, and grouping columns are required."""
    with pytest.raises(ValueError, match="lack columns"):
        aggregate_loadings(components.drop("coefficient"), "mean")
    with pytest.raises(ValueError, match="lack columns"):
        aggregate_loadings(components, "mean", by="Wavenumber")
