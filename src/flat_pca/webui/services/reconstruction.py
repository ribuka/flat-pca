"""Feature values of the reconstruction views of the exploration screen."""

from __future__ import annotations

from typing import Literal

import numpy as np

from flat_pca.feature_engineering.pca import (
    PcaModel,
    PreparedRows,
    component_contribution,
)
from flat_pca.feature_engineering.pca.reconstruct import (
    reconstruct_standardized,
    unscale,
)

ReconstructionKind = Literal["contribution", "reconstruction", "residual"]


def reconstruction_values(
    kind: ReconstructionKind,
    model: PcaModel,
    prepared: PreparedRows,
    component: int,
) -> np.ndarray:
    """Return one prepared row's contribution, reconstruction, or residual.

    Every value is in the preprocessed (intensity-transformed) feature
    space of ``X.npy``; the intensity transform is not undone.

    Parameters
    ----------
    kind : ReconstructionKind
        ``"contribution"`` for component ``component`` alone,
        ``"reconstruction"`` for the leading ``1..component`` components,
        or ``"residual"`` for the prepared values minus that
        reconstruction.
    model : PcaModel
        Fitted pipeline of the run.
    prepared : PreparedRows
        One row prepared by ``prepare_rows``; it must be kept.
    component : int
        One-based component number ``k``.

    Returns
    -------
    np.ndarray
        ``float64`` values of every feature, in ``model.columns`` order.

    Raises
    ------
    ValueError
        If the row was dropped or ``component`` is out of range.
    """
    if prepared.kept.size != 1:
        raise ValueError("exactly one kept row is required")
    if kind == "contribution":
        return component_contribution(
            prepared.scores, model.pca, model.scaling_model, model.columns, component
        )[0]
    reconstructed = unscale(
        reconstruct_standardized(prepared.scores, model.pca, component),
        model.scaling_model,
        model.columns,
    )[0]
    if kind == "reconstruction":
        return reconstructed
    return prepared.values[0] - reconstructed
