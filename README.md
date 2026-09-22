# Flat PCA

Flat PCA provides a deterministic Flatten-PCA workflow for time-series spectral
data stored in Parquet files.

## 前提

### 入力 Parquet の形式

- 同じ測定条件を持つ複数の Parquetファイルを、1 ファイルにつき 1 行の特徴量へ展開します。
- 入力パスから `source` 列を生成し、各行の識別子として使います。

## インストール

```bash
uv sync
```

## 使い方

### データ準備

```python
from pathlib import Path

input_dir = Path("data/spectra")
paths = sorted(input_dir.glob("*.parquet"))
```

### Flatten-PCA処理

```python
from polars as pl

from flat_pca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
    reshape_pca_components,
)

preprocess_cfg = {
    "target_steps": [1],
    "edge_trim": [6.0, 1.0],
    "wavelength_range": None,
    "t_smoothing_window": None,
    "w_smoothing_window": 1.0,
    "t_normalization_range": None,
    "w_normalization_range": None,
    "t_downsampling_stride": 2,
    "w_downsampling_stride": 10,
}

## 1ファイルを1行へflatten化して、全ファイルをstack
flattened: pl.LazyFrame = preprocess_and_flatten(paths, **preprocess_cfg)

## PCA を fit
pca: "PcaModel" = flatten_pca(flattened=flattened)
# pca: "PcaModel" = flatten_pca(paths, **preprocess_cfg)

## 任意の成分において寄与度が大きい特徴量を確認する(1-based)
print(pca.get_component_coefficients(1))

## flattened へスコアを結合する。
result: pl.LazyFrame = append_pca_scores(pca, flattened)

## 成分係数を long 形式の座標付きテーブルへ戻す。
components: pl.DataFrame = reshape_pca_components(pca, flattened)
```

### 可視化

- `flat_pca.visualize.create_heatmap` は、波長方向を横軸、時間方向を縦軸としたヒートマップ(`plotly.graph_objects.Figure`)を作成します。
- 入力は以下に対応してます。
    - 入力 Parquet と同じ wide 形式
    - または `reshape_pca_components` が返す long 形式
        - x_name, y_name, z_name があらかじめ存在している必要があります。

```python
from flat_pca.visualize import create_heatmap

# 入力データそのもののスペクトル強度を波長 x 時間のヒートマップとして表示する (wide形式)
fig = create_heatmap(pl.read_parquet(paths[0]))
fig.show()

# 特定の主成分の係数を、波長 × 時間 のヒートマップとして表示する (long 形式)
component_heatmap = create_heatmap(
    components.filter(pl.col("component") == 1).drop("component"),
    x_name="wavelength",
    y_name="StepTime",
    z_name="coefficient",
)
component_heatmap.show()
```
