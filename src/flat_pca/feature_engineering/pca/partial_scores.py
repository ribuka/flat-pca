"""Partial score trajectories of flattened spectral feature rows.

A row's score on a component is the sum of ``z_f w_f`` over its centered
features ``z``. Summing the features in ``(Step, Sequence, StepTime)``
order and keeping the running sum after each time point shows how the
score builds up over the run; the last point is the score itself when
the PCA does not whiten.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from flat_pca.spectral.schema import parse_feature_coordinate

from .model import PcaModel


@dataclass(frozen=True)
class PartialScores:
    """Running score sums over the time points of flattened features.

    Attributes
    ----------
    steps : np.ndarray
        ``Step`` of each time point, shaped ``(n_points,)``.
    sequences : np.ndarray
        ``Sequence`` of each time point, shaped ``(n_points,)``.
    step_times : np.ndarray
        ``StepTime`` of each time point, shaped ``(n_points,)``.
    scores : np.ndarray
        ``float64`` running sums shaped ``(n_rows, n_points, n_components)``:
        ``scores[r, p, c]`` sums ``z_f w_f`` of row ``r`` and the ``c``-th
        requested component over every feature up to time point ``p``.
    """

    steps: np.ndarray
    sequences: np.ndarray
    step_times: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class TimePointOrder:
    """Order of flattened features by ascending ``(Step, Sequence, StepTime)``.

    Attributes
    ----------
    order : np.ndarray
        Feature positions in ascending ``(Step, Sequence, StepTime)`` order.
    is_last : np.ndarray
        Boolean mask over ``order`` marking the last feature of each time
        point.
    steps : np.ndarray
        ``Step`` of each time point, shaped ``(n_points,)``.
    sequences : np.ndarray
        ``Sequence`` of each time point, shaped ``(n_points,)``.
    step_times : np.ndarray
        ``StepTime`` of each time point, shaped ``(n_points,)``.
    """

    order: np.ndarray
    is_last: np.ndarray
    steps: np.ndarray
    sequences: np.ndarray
    step_times: np.ndarray


def time_point_order(columns: Sequence[str]) -> TimePointOrder:
    """Order flattened features by their time points.

    Parameters
    ----------
    columns : Sequence[str]
        Flattened feature names.

    Returns
    -------
    TimePointOrder
        Ascending ``(Step, Sequence, StepTime)`` order of the features; the
        order of features sharing a time point is unspecified.

    Raises
    ------
    ValueError
        If a column is not a flattened feature name.
    """
    coordinates = [parse_feature_coordinate(column) for column in columns]
    steps = np.array([coordinate[1] for coordinate in coordinates], dtype=np.int64)
    sequences = np.array([coordinate[2] for coordinate in coordinates], dtype=np.int64)
    step_times = np.array([coordinate[3] for coordinate in coordinates], dtype=np.float64)
    order = np.lexsort((step_times, sequences, steps))
    # A feature ends its time point when the next one has other coordinates.
    sorted_keys = (steps[order], sequences[order], step_times[order])
    is_last = np.ones(len(order), dtype=bool)
    is_last[:-1] = np.logical_or.reduce([key[1:] != key[:-1] for key in sorted_keys])
    ends = order[is_last]
    return TimePointOrder(
        order=order,
        is_last=is_last,
        steps=steps[ends],
        sequences=sequences[ends],
        step_times=step_times[ends],
    )


def partial_scores(
    values: np.ndarray,
    model: PcaModel,
    components: Sequence[int],
    order: TimePointOrder | None = None,
) -> PartialScores:
    """Return the partial score trajectories of prepared feature rows.

    The rows are scaled with ``model.scaling_model.apply`` and centered by
    ``pca.mean_``, giving the ``z`` passed to the PCA projection. The
    features are ordered by ascending ``(Step, Sequence, StepTime)`` and the
    running sum of ``z_f w_f`` is taken in that order in ``float64``. Each
    time point keeps the sum after its last feature, so the last point
    equals the score of ``transform_pca`` when ``pca.whiten`` is false.
    Rows are computed one at a time, so the intermediate arrays do not grow
    with the number of rows.

    Parameters
    ----------
    values : np.ndarray
        Rows shaped ``(n_rows, len(model.columns))`` in the model's column
        order, already imputed and outlier-handled with the model
        (``PreparedRows.values``).
    model : PcaModel
        Fitted pipeline whose columns are flattened feature names.
    components : Sequence[int]
        One-based component numbers.
    order : TimePointOrder | None, optional
        ``time_point_order(model.columns)``, passed to reuse it across
        calls; computed when omitted.

    Returns
    -------
    PartialScores
        Time points in ascending order and the running sums of every row.

    Raises
    ------
    ValueError
        If ``values`` has the wrong shape or missing values, a component is
        out of range, or a column is not a flattened feature name.
    """
    columns = list(model.columns)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns):
        raise ValueError(
            f"values are shaped {matrix.shape}, expected (n_rows, {len(columns)})"
        )
    if np.isnan(matrix).any():
        raise ValueError("values must be imputed before computing partial scores")
    indices = [component - 1 for component in components]
    if any(not 0 <= index < model.n_component for index in indices):
        raise ValueError(
            f"components must be in 1..{model.n_component}: {list(components)}"
        )
    if order is None:
        order = time_point_order(columns)
    mean = np.asarray(model.pca.mean_, dtype=np.float64)[order.order]
    # (n_features, n_components) weights in (Step, Sequence, StepTime) order.
    weights = np.asarray(model.pca.components_, dtype=np.float64)[indices][
        :, order.order
    ].T
    scores = np.empty((matrix.shape[0], int(order.is_last.sum()), len(indices)))
    for row in range(matrix.shape[0]):
        scaled = (
            model.scaling_model.apply(
                pl.from_numpy(matrix[row : row + 1], schema=columns, orient="row").lazy(),
                columns,
            )
            .select(columns)
            .collect()
            .to_numpy()[0]
            .astype(np.float64)
        )
        centered = scaled[order.order] - mean
        running = np.cumsum(centered[:, np.newaxis] * weights, axis=0)
        scores[row] = running[order.is_last]
    return PartialScores(
        steps=order.steps,
        sequences=order.sequences,
        step_times=order.step_times,
        scores=scores,
    )
