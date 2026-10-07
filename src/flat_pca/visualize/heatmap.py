"""Spectral heatmap rendering."""

from collections.abc import Collection, Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from loguru import logger

from flat_pca.spectral.schema import (
    NON_SPECTRAL_COLUMNS as SPECTRAL_NON_SPECTRAL_COLUMNS,
)

NON_SPECTRAL_COLUMNS = frozenset(SPECTRAL_NON_SPECTRAL_COLUMNS)
# Values centered on zero, drawn with a zero-centered diverging color scale.
DIVERGING_VALUE_NAMES = frozenset({"coefficient", "contribution", "residual"})


def _symmetric_color_range(z: np.ndarray, quantile: float) -> tuple[float, float]:
    """Compute a zero-centered color range from a quantile of ``abs(z)``.

    Useful for diverging data (e.g. PCA coefficients) where a symmetric
    range around zero avoids a few outlier values washing out the scale.
    """
    vmax = float(np.nanquantile(np.abs(z), quantile))
    return -vmax, vmax


def create_heatmap(
    spectra: pl.DataFrame | None = None,
    *,
    z: np.ndarray | None = None,
    x: Sequence[float] | np.ndarray | None = None,
    y: Sequence[float] | np.ndarray | None = None,
    x_name: str = "wavelength",
    y_name: str = "Time",
    z_name: str = "intensity",
    strip_suffix: str | None = "nm",
    metadata_columns: Collection[str] = NON_SPECTRAL_COLUMNS,
    color_continuous_scale: str | None = None,
    range_color: tuple[float, float] | list[float] | None = None,
    symmetric_range_quantile: float = 0.995,
) -> go.Figure:
    """Create a spectral-intensity heatmap over time and wavelength.

    The data are given either as ``spectra`` or as a matrix ``z`` with its
    axes ``x`` and ``y``.

    Parameters
    ----------
    spectra : pl.DataFrame | None, optional
        Either wide-format spectral data with ``Time``, ``Step``, and
        ``Sequence`` metadata columns (and optionally ``StepTime`` and
        ``ReverseStepTime``), where every other column name is a numeric
        wavelength with an optional ``"nm"`` suffix, or already
        long-format data containing ``x_name``, ``y_name``, and ``z_name``
        columns. ``None`` (default) when ``z``, ``x``, and ``y`` are given.
    z : np.ndarray | None, optional
        Matrix shaped ``(len(y), len(x))`` used instead of ``spectra``.
        NaN cells are drawn blank.
    x, y : Sequence[float] | np.ndarray | None, optional
        Ascending wavelength and time axes of ``z``.
    x_name, y_name, z_name : str, optional
        Long-format column names for wavelength, time, and intensity, by
        default ``"wavelength"``, ``"Time"``, and ``"intensity"``.
    metadata_columns : Collection[str], optional
        Non-spectral columns (metadata and StepTime columns) to exclude from
        the wavelength columns of wide-format data, by default
        ``NON_SPECTRAL_COLUMNS``.
    strip_suffix : str | None, optional
        Suffix to strip from ``x_name`` values before casting to float, by
        default ``"nm"``. If ``None``, the column is left as-is.
    color_continuous_scale : str | None, optional
        Plotly color scale name. If ``None`` (default), it is chosen
        automatically based on ``z_name``: ``"RdBu_r"`` when
        ``z_name`` is in ``DIVERGING_VALUE_NAMES``, otherwise ``"Viridis"``.
    range_color : tuple[float, float] | list[float] | None, optional
        Explicit color-scale limits. If ``None`` (default), limits are
        derived from the data.
    symmetric_range_quantile : float, optional
        Quantile of ``abs(z)`` used to derive a zero-centered
        ``range_color`` when ``z_name`` is in ``DIVERGING_VALUE_NAMES`` and
        ``range_color`` is not explicitly set, by default ``0.995``.

    Returns
    -------
    go.Figure
        Heatmap with wavelength on the x-axis and time on the y-axis, with
        time zero displayed at the bottom.

    Raises
    ------
    ValueError
        If both or neither of ``spectra`` and the matrix are given, or the
        matrix shape does not match its axes.
    """
    matrix = (z, x, y)
    if spectra is None:
        if any(item is None for item in matrix):
            raise ValueError("give either spectra or all of z, x, and y")
        z_values = np.asarray(z, dtype=np.float64)
        x_values = [float(value) for value in x]  # type: ignore[union-attr]
        y_values = [float(value) for value in y]  # type: ignore[union-attr]
        if z_values.shape != (len(y_values), len(x_values)):
            raise ValueError(
                f"z is shaped {z_values.shape}, expected {(len(y_values), len(x_values))}"
            )
    elif any(item is not None for item in matrix):
        raise ValueError("give either spectra or all of z, x, and y")
    else:
        z_values, x_values, y_values = _frame_matrix(
            spectra,
            x_name=x_name,
            y_name=y_name,
            z_name=z_name,
            strip_suffix=strip_suffix,
            metadata_columns=metadata_columns,
        )

    is_diverging = z_name in DIVERGING_VALUE_NAMES
    if color_continuous_scale is None:
        color_continuous_scale = "RdBu_r" if is_diverging else "Viridis"
    if range_color is None and is_diverging:
        range_color = _symmetric_color_range(z_values, symmetric_range_quantile)

    return px.imshow(
        z_values,
        x=x_values,
        y=y_values,
        origin="lower",
        aspect="auto",
        color_continuous_scale=color_continuous_scale,
        labels={"x": x_name, "y": y_name, "color": z_name},
        range_color=range_color,
    ).update_xaxes(
        tickangle=-60
    )


def _frame_matrix(
    spectra: pl.DataFrame,
    *,
    x_name: str,
    y_name: str,
    z_name: str,
    strip_suffix: str | None,
    metadata_columns: Collection[str],
) -> tuple[np.ndarray, list[float], list[object]]:
    """Pivot wide- or long-format spectra into a heatmap matrix.

    Parameters
    ----------
    spectra : pl.DataFrame
        Wide- or long-format spectral data (see ``create_heatmap``).
    x_name, y_name, z_name : str
        Long-format column names for wavelength, time, and intensity.
    strip_suffix : str | None
        Suffix stripped from wavelength names before casting to float.
    metadata_columns : Collection[str]
        Non-spectral columns of wide-format data.

    Returns
    -------
    tuple[np.ndarray, list[float], list[object]]
        Matrix shaped ``(time, wavelength)``, and its ascending wavelength
        and time axes.
    """
    if {x_name, y_name, z_name}.issubset(spectra.columns):
        logger.debug("Using long-format data as-is.")
        spectra_long = spectra
    else:
        x_columns = [c for c in spectra.columns if c not in metadata_columns]
        spectra_long = spectra.unpivot(
            on=x_columns,
            index=y_name,
            variable_name=x_name,
            value_name=z_name,
        )
        if strip_suffix is not None:
            spectra_long = spectra_long.with_columns(
                pl.col(x_name).str.strip_suffix(strip_suffix).cast(pl.Float64)
            )
    spectra_long = spectra_long.sort(x_name)
    heatmap_data = spectra_long.pivot(
        on=x_name,
        index=y_name,
        values=z_name,
        aggregate_function="first",
    ).sort(y_name)

    return (
        heatmap_data.drop(y_name).to_numpy(),
        [float(column) for column in heatmap_data.columns[1:]],
        heatmap_data[y_name].to_list(),
    )
