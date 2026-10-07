"""Scree plot of the explained variance of PCA components."""

import plotly.graph_objects as go
import polars as pl


def create_scree_plot(table: pl.DataFrame) -> go.Figure:
    """Create a scree plot of the explained-variance ratio of each component.

    Parameters
    ----------
    table : pl.DataFrame
        Result of ``PcaModel.get_explained_variance_table``, with the
        ``component``, ``explained_variance_ratio``, and
        ``cumulative_explained_variance`` columns.

    Returns
    -------
    go.Figure
        Bars of each component's ratio and a line of the cumulative ratio,
        both on one axis from 0 to 1.
    """
    components = table["component"].to_numpy()
    return go.Figure(
        [
            go.Bar(
                x=components,
                y=table["explained_variance_ratio"].to_numpy(),
                name="explained_variance_ratio",
            ),
            go.Scatter(
                x=components,
                y=table["cumulative_explained_variance"].to_numpy(),
                mode="lines+markers",
                name="cumulative_explained_variance",
            ),
        ]
    ).update_layout(
        xaxis_title="component",
        yaxis_title="explained variance ratio",
        yaxis_range=[0, 1.05],
        showlegend=True,
    )
