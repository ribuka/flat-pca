"""Preprocessing settings of a fit configuration, as the fit form saves them."""

from __future__ import annotations


def preprocess_settings(**overrides: object) -> dict[str, object]:
    """Return every ``preprocess`` setting the fit form writes.

    Parameters
    ----------
    **overrides : object
        Settings replacing the defaults.

    Returns
    -------
    dict[str, object]
        Steps 1 and 2 with no optional stage, no intensity transform, no
        downsampling, and a missing ratio of at most 0.1, updated with
        ``overrides``.
    """
    return {
        "target_steps": [1, 2],
        "edge_trim": None,
        "wavelength_range": None,
        "t_normalization_range": None,
        "w_normalization_range": None,
        "t_smoothing_window": None,
        "w_smoothing_window": None,
        "intensity_transform": "none",
        "intensity_transform_scale": 1.0,
        "t_downsampling_stride": 1,
        "w_downsampling_stride": 1,
        "max_null_ratio": 0.1,
        **overrides,
    }
