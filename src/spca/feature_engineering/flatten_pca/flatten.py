"""Deterministic spectral flattening for Flatten-PCA."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .schema import parse_wavelength, wavelength_columns


def flatten_inputs(inputs: Sequence[tuple[Path, pl.DataFrame]]) -> pl.DataFrame:
    """Flatten each validated spectral frame into one deterministic row.

    Parameters
    ----------
    inputs : Sequence[tuple[Path, pl.DataFrame]]
        Normalized input paths paired with validated spectral frames.

    Returns
    -------
    pl.DataFrame
        One row per input file, ordered by normalized path, with ``filename``
        followed by spectral features ordered by numeric wavelength, Step,
        Sequence, and Time.

    Raises
    ------
    ValueError
        If no inputs are provided, formatted feature names collide, or frames
        do not produce the same ordered feature names.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must contain at least one validated frame")

    rows: list[dict[str, object]] = []
    expected_feature_names: list[str] | None = None
    for path, frame in sorted(inputs, key=lambda item: str(item[0].resolve())):
        spectra = sorted(
            (
                (parse_wavelength(column), column)
                for column in wavelength_columns(frame.columns)
            ),
            key=lambda item: item[0],
        )
        sorted_frame = frame.sort("Step", "Sequence", "Time")
        feature_names: list[str] = []
        feature_values: list[object] = []
        for _, wavelength_column in spectra:
            for step, sequence, time, intensity in sorted_frame.select(
                "Step",
                "Sequence",
                "Time",
                wavelength_column,
            ).iter_rows():
                feature_names.append(
                    f"{wavelength_column}_{int(step)}_{int(sequence)}_{float(time):.2f}"
                )
                feature_values.append(intensity)

        if len(feature_names) != len(set(feature_names)):
            raise ValueError("duplicate flattened feature names")
        if expected_feature_names is None:
            expected_feature_names = feature_names
        elif feature_names != expected_feature_names:
            raise ValueError("input frames must produce matching flattened features")

        row: dict[str, object] = {"filename": path.stem}
        row.update(zip(feature_names, feature_values, strict=True))
        rows.append(row)

    return pl.DataFrame(rows).select("filename", *expected_feature_names)
