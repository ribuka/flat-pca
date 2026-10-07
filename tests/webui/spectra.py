"""Synthetic spectra files for fit-job tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

SPECTRA_WAVELENGTHS = (400.0, 401.0, 402.5, 405.0, 410.0)
SPECTRA_FILE_COUNT = 12
# Index of the file lacking the last row of Step 2.
SPECTRA_SHORT_FILE = 3


def write_spectra(directory: Path) -> list[Path]:
    """Write synthetic spectra files.

    Each file holds ``Step`` 1 (four rows) and ``Step`` 2 (three rows) of
    one ``Sequence`` over ``SPECTRA_WAVELENGTHS``, from two latent factors
    plus noise. File ``SPECTRA_SHORT_FILE`` lacks the last ``Step`` 2 row,
    so its flattened features of that time point are missing.

    Parameters
    ----------
    directory : Path
        Directory to create and write into.

    Returns
    -------
    list[Path]
        ``s-{index:02d}.parquet`` paths in index order.
    """
    directory.mkdir(parents=True)
    rng = np.random.default_rng(11)
    loadings = rng.normal(size=(2, 7 * len(SPECTRA_WAVELENGTHS)))
    paths = []
    for index in range(SPECTRA_FILE_COUNT):
        values = (rng.normal(size=2) @ loadings).reshape(7, len(SPECTRA_WAVELENGTHS))
        values += 5.0 + 0.05 * rng.normal(size=values.shape)
        frame = pl.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Step": [1, 1, 1, 1, 2, 2, 2],
                "Sequence": [1] * 7,
                **{
                    f"{wavelength:.1f}nm": values[:, column]
                    for column, wavelength in enumerate(SPECTRA_WAVELENGTHS)
                },
            }
        )
        if index == SPECTRA_SHORT_FILE:
            frame = frame.head(6)
        path = directory / f"s-{index:02d}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths
