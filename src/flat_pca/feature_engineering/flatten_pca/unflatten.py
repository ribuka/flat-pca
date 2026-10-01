"""Convert flattened spectral features back to coordinate-based layouts."""

from __future__ import annotations

import polars as pl

from flat_pca.spectral.schema import (
    SOURCE_COLUMN,
    flattened_feature_columns,
    parse_feature_coordinate,
)

_ROW = "__flatten_source_row"
_FEATURE = "__flatten_feature"
_COORDINATES = ("Step", "Sequence", "StepTime", "wavelength")


def _feature_layout(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Validate feature names and return their coordinate lookup table.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened frame with a source column.

    Returns
    -------
    pl.DataFrame
        Feature names and decoded numeric coordinates.

    Raises
    ------
    ValueError
        If source or valid flattened features are missing, or another
        non-feature column is present.
    """
    columns = frame.collect_schema().names()
    if SOURCE_COLUMN not in columns:
        raise ValueError("flattened frame must contain a source column")
    features = flattened_feature_columns(columns)
    if not features:
        raise ValueError("flattened frame must contain feature columns")
    coordinates = [parse_feature_coordinate(column) for column in features]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("flattened feature coordinates must be unique")
    return pl.DataFrame(
        {
            _FEATURE: features,
            "wavelength": [item[0] for item in coordinates],
            "Step": [item[1] for item in coordinates],
            "Sequence": [item[2] for item in coordinates],
            "StepTime": [item[3] for item in coordinates],
        },
        schema_overrides={
            "wavelength": pl.Float64,
            "Step": pl.Int64,
            "Sequence": pl.Int64,
            "StepTime": pl.Float64,
        },
    )


def _as_long(
    frame: pl.DataFrame | pl.LazyFrame,
    layout: pl.DataFrame,
    value_name: str,
) -> pl.LazyFrame:
    """Build a lazy long query with a private source-row identifier.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened spectral features.
    layout : pl.DataFrame
        Validated feature coordinate lookup table.
    value_name : str
        Output value column name.

    Returns
    -------
    pl.LazyFrame
        Long query retaining source-row identity for wide reconstruction.
    """
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return (
        lazy.with_row_index(_ROW)
        .unpivot(
            on=layout[_FEATURE].to_list(),
            index=[_ROW, SOURCE_COLUMN],
            variable_name=_FEATURE,
            value_name=value_name,
        )
        .join(layout.lazy(), on=_FEATURE, how="left")
        .drop(_FEATURE)
    )


def flatten_to_long(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    value_name: str = "intensity",
) -> pl.DataFrame | pl.LazyFrame:
    """Restore a flattened frame to long spectral coordinates.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened features with ``source``. Every other column must have a
        canonical flattened feature name; extra metadata columns are rejected.
    value_name : str, optional
        Name of the value column, by default ``"intensity"``.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Source, Step, Sequence, StepTime, wavelength, and spectral value.
        Missing grid values remain null. The output has the input frame type.
    """
    if value_name in {SOURCE_COLUMN, *_COORDINATES, _ROW, _FEATURE}:
        raise ValueError(f"value column conflicts with coordinate column: {value_name!r}")
    layout = _feature_layout(frame)
    result = _as_long(frame, layout, value_name).select(
        SOURCE_COLUMN, "Step", "Sequence", "StepTime", "wavelength", value_name
    )
    return result.collect() if isinstance(frame, pl.DataFrame) else result


def flatten_to_wide(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    drop_all_null_rows: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Restore a flattened frame to source and grid rows with wavelength columns.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened features with ``source``. Extra non-feature columns are
        rejected.
    drop_all_null_rows : bool, optional
        Remove grid rows where every wavelength value is null, by default
        ``False``. This can remove rows absent from a source's original grid.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Rows sorted by source, Step, Sequence, and StepTime, with wavelength
        columns in numeric order. The output has the input frame type.
    """
    layout = _feature_layout(frame)
    wavelengths = sorted(layout["wavelength"].unique().to_list())
    wavelength_names = [f"{wavelength:.1f}nm" for wavelength in wavelengths]
    long = _as_long(frame, layout, "__flatten_value")
    # A lazy pivot cannot infer its output schema from data. The validated
    # feature names give us the wavelength columns before executing the query.
    result = long.group_by(_ROW, SOURCE_COLUMN, "Step", "Sequence", "StepTime").agg(
        pl.col("__flatten_value")
        .filter(pl.col("wavelength") == wavelength)
        .first()
        .alias(column)
        for wavelength, column in zip(wavelengths, wavelength_names, strict=True)
    )
    if drop_all_null_rows:
        result = result.filter(pl.any_horizontal(pl.col(wavelength_names).is_not_null()))
    result = result.sort(SOURCE_COLUMN, "Step", "Sequence", "StepTime", _ROW).select(
        SOURCE_COLUMN, "Step", "Sequence", "StepTime", *wavelength_names
    )
    return result.collect() if isinstance(frame, pl.DataFrame) else result
