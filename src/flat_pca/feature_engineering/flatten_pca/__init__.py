"""Public Flatten-PCA API."""

from .api import (
    flatten_pca,
    materialize_and_drop_sparse_feature_columns,
    materialize_flattened,
    preprocess_and_flatten,
)
from .component_reshape import reshape_pca_components
from .pca_scores import append_pca_scores

__all__ = [
    "append_pca_scores",
    "flatten_pca",
    "materialize_and_drop_sparse_feature_columns",
    "materialize_flattened",
    "preprocess_and_flatten",
    "reshape_pca_components",
]
