# SPCA

SPCA provides a deterministic Flatten-PCA workflow for time-series spectral
data stored in Parquet files.

## Minimal Python API example

```python
from pathlib import Path

from spca.feature_engineering import flatten_pca

paths = sorted(Path("tests/fixtures/real_subset").glob("*.parquet"))
result = flatten_pca(paths, n_component=2)
print(result)
```

Pass one or more paths with unique filename stems. `n_component` must be an
integer from 1 through the smaller of the number of files and the number of
flattened spectral features.

## Input schema

Each Parquet file must contain numeric `Time`, `Step`, and `Sequence` columns.
Every other column is a numeric spectrum whose name uses the canonical
one-decimal wavelength form, such as `350.0nm`. Metadata and spectral values
must be finite and non-null, `(Time, Step, Sequence)` must be unique within each
file, and all files must contain the same metadata keys and wavelengths.

## Optional preprocessing

The keyword arguments enable preprocessing in this fixed order:

1. `t_smoothing_window`: centered, closed time-window smoothing.
2. `w_smoothing_window`: centered, closed wavelength-window smoothing.
3. `t_normalization_range`: normalization over an inclusive time range.
4. `w_normalization_range`: normalization over an inclusive wavelength range.

Omit an argument or pass `None` to disable that stage. Window widths and range
bounds use the same units as `Time` and wavelength respectively.

## Output

`flatten_pca` returns a Polars `DataFrame` with one row per input file. Columns
are ordered as `filename`, deterministic flattened spectral features, then
`pca-1` through `pca-{n_component}`. Input files and features are sorted
deterministically, so output order does not depend on the order of `paths`.
