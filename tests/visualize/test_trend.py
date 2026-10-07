"""Tests for the trend line plot."""

import numpy as np

from flat_pca.visualize import create_trend


def test_create_trend_draws_one_trace_per_label() -> None:
    """Each labeled series becomes a named line in mapping order."""
    figure = create_trend(
        {
            "run-1": ([0.0, 1.0], [2.0, np.nan]),
            "run-2": (np.array([0.0, 2.0]), np.array([3.0, 4.0])),
        },
        x_name="StepTime",
        title="wavelength = 400",
    )

    assert [trace.name for trace in figure.data] == ["run-1", "run-2"]
    assert list(figure.data[1].x) == [0.0, 2.0]
    assert np.isnan(figure.data[0].y[1])
    assert figure.layout.xaxis.title.text == "StepTime"
    assert figure.layout.yaxis.title.text == "intensity"
    assert figure.layout.title.text == "wavelength = 400"
