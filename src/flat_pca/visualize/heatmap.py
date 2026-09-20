import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

METADATA_COLUMNS = {"Time", "Step", "Sequence"}


def _symmetric_color_range(z: np.ndarray, quantile: float) -> tuple[float, float]:
    """Compute a zero-centered color range from a quantile of ``abs(z)``.

    Useful for diverging data (e.g. PCA coefficients) where a symmetric
    range around zero avoids a few outlier values washing out the scale.
    """
    vmax = float(np.nanquantile(np.abs(z), quantile))
    return -vmax, vmax


def create_heatmap(
    spectra: pl.DataFrame,
    *,
    x_name: str = "wavelength",
    y_name: str = "Time",
    z_name: str = "intensity",
    strip_suffix: str | None = "nm",
    metadata_columns: set[str] = METADATA_COLUMNS,
    color_continuous_scale: str | None = None,
    range_color: tuple[float, float] | list[float] | None = None,
    symmetric_range_quantile: float = 0.995,
) -> go.Figure:
    """Create a spectral-intensity heatmap over time and wavelength.

    Parameters
    ----------
    spectra : pl.DataFrame
        Either wide-format spectral data with ``Time``, ``Step``, and
        ``Sequence`` metadata columns, where every other column name is a
        numeric wavelength with an optional ``"nm"`` suffix, or already
        long-format data containing ``x_name``, ``y_name``, and ``z_name``
        columns.

    Returns
    -------
    go.Figure
        Heatmap with wavelength on the x-axis and time on the y-axis.

    metadata_columns : set[str], optional
        Set of metadata columns to exclude from the spectral data, by default METADATA_COLUMNS
        Time zero is displayed at the bottom.
    strip_suffix : str | None, optional
        Suffix to strip from ``x_name`` values before casting to float, by
        default ``"nm"``. If ``None``, the column is left as-is.
    color_continuous_scale : str | None, optional
        Plotly color scale name. If ``None`` (default), it is chosen
        automatically based on ``z_name``: ``"RdBu_r"`` when
        ``z_name == "coefficient"``, otherwise ``"Viridis"``.
    symmetric_range_quantile : float, optional
        Quantile of ``abs(z)`` used to derive a zero-centered
        ``range_color`` when ``z_name == "coefficient"`` and
        ``range_color`` is not explicitly set, by default ``0.995``.
    """
    if {x_name, y_name, z_name}.issubset(spectra.columns):
        print("Using long-format data as-is.")
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

    z = heatmap_data.drop(y_name).to_numpy()

    is_coefficient = z_name == "coefficient"
    if color_continuous_scale is None:
        color_continuous_scale = "RdBu_r" if is_coefficient else "Viridis"
    if range_color is None and is_coefficient:
        range_color = _symmetric_color_range(z, symmetric_range_quantile)

    return px.imshow(
        z,
        x=[float(column) for column in heatmap_data.columns[1:]],
        y=heatmap_data[y_name].to_list(),
        origin="lower",
        aspect="auto",
        color_continuous_scale=color_continuous_scale,
        labels={"x": x_name, "y": y_name, "color": z_name},
        range_color=range_color,
    ).update_xaxes(
        tickangle=-60
    )
