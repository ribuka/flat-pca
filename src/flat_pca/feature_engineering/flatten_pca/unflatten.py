"""Convert flattened spectral features back to coordinate-based layouts."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import polars as pl

from flat_pca.spectral.schema import (
    SOURCE_COLUMN,
    flattened_feature_columns,
    parse_feature_coordinate,
)

_ROW = "__flatten_source_row"
_FEATURE = "__flatten_feature"
_VALUE = "__flatten_value"
_COORDINATES = ("Step", "Sequence", "StepTime", "wavelength")
_RESERVED = frozenset({SOURCE_COLUMN, *_COORDINATES, _ROW, _FEATURE, _VALUE})


def _resolve_index(
    columns: Sequence[str],
    index: str | Sequence[str] | None,
) -> list[str]:
    """Normalize and validate the columns retained alongside features.

    Parameters
    ----------
    columns : Sequence[str]
        Complete column names of the flattened frame.
    index : str | Sequence[str] | None
        Column name or names to retain, or ``None`` for none.

    Returns
    -------
    list[str]
        Index column names in the requested order.

    Raises
    ------
    ValueError
        If names are duplicated, missing from the frame, or conflict with
        source, coordinate, or internal working columns.
    """
    if index is None:
        return []
    names = [index] if isinstance(index, str) else list(index)
    if len(set(names)) != len(names):
        raise ValueError(f"index column names must be unique: {names!r}")
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"index columns not found in flattened frame: {missing!r}")
    _reject_index_conflicts(names, _RESERVED)
    return names


def _reject_index_conflicts(index: Sequence[str], reserved: Collection[str]) -> None:
    """Reject index columns whose names are used by the output layout.

    Parameters
    ----------
    index : Sequence[str]
        Index column names.
    reserved : Collection[str]
        Column names produced or used internally by the conversion.

    Raises
    ------
    ValueError
        If an index column name is in ``reserved``.
    """
    conflicts = [name for name in index if name in reserved]
    if conflicts:
        raise ValueError(f"index columns conflict with output columns: {conflicts!r}")


def _feature_layout(columns: Sequence[str], index: Sequence[str]) -> pl.DataFrame:
    """Validate feature names and return their coordinate lookup table.

    Parameters
    ----------
    columns : Sequence[str]
        Complete column names of the flattened frame.
    index : Sequence[str]
        Validated index column names, which are not treated as features.

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
    if SOURCE_COLUMN not in columns:
        raise ValueError("flattened frame must contain a source column")
    retained = set(index)
    features = [
        column
        for column in flattened_feature_columns(list(columns))
        if column not in retained
    ]
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
    index: Sequence[str],
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
    index : Sequence[str]
        Validated columns repeated on every long row from the same input row.

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
            index=[_ROW, SOURCE_COLUMN, *index],
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
    index: str | Sequence[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Restore a flattened frame to long spectral coordinates.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened features with ``source``. Every other column must have a
        canonical flattened feature name or be listed in ``index``; other
        metadata columns are rejected.
    value_name : str, optional
        Name of the value column, by default ``"intensity"``.
    index : str | Sequence[str] | None, optional
        Columns to retain, by default ``None``. As with the ``index`` of
        ``polars.DataFrame.unpivot``, each value is repeated on every long row
        expanded from its input row.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Source, index columns in the given order, Step, Sequence, StepTime,
        wavelength, and spectral value. Missing grid values remain null. The
        output has the input frame type.

    Raises
    ------
    ValueError
        If ``value_name`` or ``index`` conflicts with another output column,
        ``index`` is invalid, or the frame has other non-feature columns.
    """
    if value_name in _RESERVED:
        raise ValueError(f"value column conflicts with coordinate column: {value_name!r}")
    columns = frame.collect_schema().names()
    index_columns = _resolve_index(columns, index)
    _reject_index_conflicts(index_columns, {value_name})
    layout = _feature_layout(columns, index_columns)
    result = _as_long(frame, layout, value_name, index_columns).select(
        SOURCE_COLUMN,
        *index_columns,
        "Step",
        "Sequence",
        "StepTime",
        "wavelength",
        value_name,
    )
    return result.collect() if isinstance(frame, pl.DataFrame) else result


def flatten_to_wide(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    drop_all_null_rows: bool = False,
    index: str | Sequence[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Restore a flattened frame to source and grid rows with wavelength columns.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Flattened features with ``source``. Non-feature columns other than
        those listed in ``index`` are rejected.
    drop_all_null_rows : bool, optional
        Remove grid rows where every wavelength value is null, by default
        ``False``. This can remove rows absent from a source's original grid.
        Index values are not considered.
    index : str | Sequence[str] | None, optional
        Columns to retain, by default ``None``. As with the ``index`` of
        ``polars.DataFrame.unpivot``, each value is repeated on every grid row
        expanded from its input row. Index columns are not used as grouping
        keys, so non-groupable dtypes such as lists are supported.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Rows sorted by source, Step, Sequence, and StepTime. Columns are
        source, index columns in the given order, Step, Sequence, StepTime,
        and wavelength columns in numeric order. The output has the input
        frame type.

    Raises
    ------
    ValueError
        If ``index`` is invalid or conflicts with an output column, or the
        frame has other non-feature columns.
    """
    columns = frame.collect_schema().names()
    index_columns = _resolve_index(columns, index)
    layout = _feature_layout(columns, index_columns)
    wavelengths = sorted(layout["wavelength"].unique().to_list())
    wavelength_names = [f"{wavelength:.1f}nm" for wavelength in wavelengths]
    _reject_index_conflicts(index_columns, wavelength_names)
    long = _as_long(frame, layout, _VALUE, index_columns)
    # A lazy pivot cannot infer its output schema from data. The validated
    # feature names give us the wavelength columns before executing the query.
    # Each group comes from one input row, so the first index value is that
    # row's value.
    result = long.group_by(_ROW, SOURCE_COLUMN, "Step", "Sequence", "StepTime").agg(
        *(pl.col(name).first() for name in index_columns),
        *(
            pl.col(_VALUE)
            .filter(pl.col("wavelength") == wavelength)
            .first()
            .alias(column)
            for wavelength, column in zip(wavelengths, wavelength_names, strict=True)
        ),
    )
    if drop_all_null_rows:
        result = result.filter(pl.any_horizontal(pl.col(wavelength_names).is_not_null()))
    result = result.sort(SOURCE_COLUMN, "Step", "Sequence", "StepTime", _ROW).select(
        SOURCE_COLUMN, *index_columns, "Step", "Sequence", "StepTime", *wavelength_names
    )
    return result.collect() if isinstance(frame, pl.DataFrame) else result
