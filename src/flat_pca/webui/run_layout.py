"""File names and configuration keys of fit and transform run directories.

The fit and transform jobs write these files and the display services read
them; ``docs/spec/webui.md`` describes their contents.
"""

from __future__ import annotations

FEATURES_FILE = "features.parquet"
SAMPLES_FILE = "samples.parquet"
X_FILE = "X.npy"
COMPONENTS_FILE = "components.npy"
PCA_STATE_FILE = "pca_state.npz"
SCORES_FILE = "scores.parquet"

# A transform run's configuration names the fit run whose model it uses.
FIT_RUN_ID_KEY = "fit_run_id"
FIT_RUN_DIR_KEY = "fit_run_dir"
