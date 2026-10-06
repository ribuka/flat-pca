"""Positionally aligned per-row data of one file in the NumPy fast path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FileRows:
    """One file's per-row metadata and values, kept positionally aligned.

    Every row reduction goes through ``take`` so that ``times``, ``steps``,
    ``sequences``, ``step_times``, and ``values`` can never drift apart.

    Parameters
    ----------
    times : np.ndarray
        Per-row Time values (float64).
    steps : list[object]
        Per-row Step values.
    sequences : list[object]
        Per-row Sequence values.
    step_times : np.ndarray
        Per-row StepTime values.
    values : np.ndarray
        ``(row, wavelength)`` value matrix.

    Raises
    ------
    ValueError
        If the fields do not all have the same row count.
    """

    times: np.ndarray
    steps: list[object]
    sequences: list[object]
    step_times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """Reject fields whose row counts differ."""
        row_counts = {
            len(self.times),
            len(self.steps),
            len(self.sequences),
            len(self.step_times),
            self.values.shape[0],
        }
        if len(row_counts) != 1:
            raise ValueError("FileRows fields must all have the same row count")

    def take(self, positions: np.ndarray, *, values: np.ndarray | None = None) -> FileRows:
        """Return the rows at ``positions``, in that order.

        Parameters
        ----------
        positions : np.ndarray
            Integer row positions to keep.
        values : np.ndarray | None, default None
            Value matrix already computed for exactly the rows at
            ``positions``, used instead of ``self.values[positions]``. Lets a
            stage that computes values only at the kept rows skip copying
            the original rows first.

        Returns
        -------
        FileRows
            The kept rows, with every field reduced together.

        Raises
        ------
        ValueError
            If ``values`` does not have one row per position.
        """
        position_list = positions.tolist()
        return FileRows(
            times=self.times[positions],
            steps=[self.steps[index] for index in position_list],
            sequences=[self.sequences[index] for index in position_list],
            step_times=self.step_times[positions],
            values=self.values[positions] if values is None else values,
        )
