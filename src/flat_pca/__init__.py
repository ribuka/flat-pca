"""Public Flatten-PCA package API."""

from .feature_engineering.flatten_pca import flatten_to_long, flatten_to_wide

__all__ = ["flatten_to_long", "flatten_to_wide"]
