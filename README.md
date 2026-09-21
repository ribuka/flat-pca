# Flat PCA

Flat PCA provides a deterministic Flatten-PCA workflow for time-series spectral
data stored in Parquet files.

## 使い方

同じ測定条件（`Time`、`Step`、`Sequence`、波長列）を持つ複数の Parquet
ファイルを、1 ファイルにつき 1 行の特徴量へ展開します。各行を識別する `source` 列は
入力パスから決まります。公開 API は、遅延した前処理・flatten、PCA の fit、スコアの
結合、成分の reshape をそれぞれ分離しています。

まず、パッケージをインストールします。

```bash
uv sync
```

次の例では、`data/spectra` にある Parquet ファイルすべてを入力にし、第 1・第 2
主成分を計算します。

```python
from pathlib import Path

from polars as pl

from flat_pca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
    reshape_pca_components,
)

input_dir = Path("data/spectra")
paths = sorted(input_dir.glob("*.parquet"))

## 前処理済みの 1 ファイル 1 行の特徴量を、必要な時点で collect する。
flattened: pl.LazyFrame = preprocess_and_flatten(paths)

## PCA を fit
pca: "PcaModel" = flatten_pca(flattened=flattened)
# pca: "PcaModel" = flatten_pca(paths)

## flattened へスコアを結合する。
result: pl.LazyFrame = append_pca_scores(pca, flattened)
result = result.collect()

## 任意の成分において寄与度が大きい特徴量を確認する(1-based)
print(pca.get_component_coefficients(1))

## 成分係数を long 形式の座標付きテーブルへ戻す。
components: pl.DataFrame = reshape_pca_components(pca, flattened)
```

`paths` には 1 個以上のパスを渡します。各行を識別する `source` 列は正規化した入力パスの
`path.as_posix()` を使うため、拡張子を除いたファイル名(stem)が重複していても既定では
問題ありません。stem の重複を警告またはエラーにしたい場合は `stem_uniqueness="warn"` /
`"error"` を指定します(既定値は `"skip"`)。
`n_component` を省略すると、「入力ファイル数」と「展開後のスペクトル特徴量数」の小さい方まで
すべての主成分を計算します。必要な主成分数だけに絞る場合は、1 以上かつこの上限以下の整数を
指定します。入力パスの順序は結果に影響しません。

### 前処理を指定する例

必要に応じて、`Step` によるフィルタ、セグメント端のトリム、波長範囲によるフィルタ、
時間・波長方向の平滑化と規格化を同時に指定できます。窓幅は半窓幅で、範囲の両端を含みます。
単位はそれぞれ入力の `Time`(トリムは `StepTime`/`ReverseStepTime`)と波長列名の数値部分に
合わせます。`wavelength_range` を指定すると、その閉区間に含まれる波長列だけを残し、
範囲外の波長列は以降の平滑化・規格化・間引きの対象から除きます。

```python
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

flattened = preprocess_and_flatten(paths, **preprocess_cfg)
pca = flatten_pca(paths, **preprocess_cfg)
```

### 可視化

`flat_pca.visualize.create_heatmap` は、波長方向を横軸、時間方向を縦軸としたヒート
マップ(`plotly.graph_objects.Figure`)を作成します。入力 Parquet と同じワイド形式
(`Time`/`Step`/`Sequence` と波長列)、または `reshape_pca_components` が返す long 形式
のどちらも渡せます。

```python
from flat_pca.visualize import create_heatmap

# 入力データそのもの(ワイド形式)をスペクトル強度のヒートマップとして表示する
create_heatmap(pl.read_parquet(paths[0]))

# 特定の主成分の係数を、wavelength × StepTime のヒートマップとして表示する(long 形式)
create_heatmap(
    components.filter(pl.col("component") == 0).drop("component"),
    x_name="wavelength",
    y_name="StepTime",
    z_name="coefficient",
)
```

## 入力 Parquet の形式

各 Parquet ファイルには数値型の `Time`、`Step`、`Sequence` 列が必要です。その他の列は
スペクトル値の数値列とし、列名は `350.0nm` のような小数第 1 位までの波長形式にします。
値は欠損・NaN・無限大を含めず、各ファイル内の `(Time, Step, Sequence)` は一意である必要が
あります。また、すべての入力ファイルでメタデータの組み合わせと波長列の集合を一致させます。

## 出力

`preprocess_and_flatten` の戻り値は Polars の `LazyFrame` です。collect 後は 1 行が入力
ファイル 1 個に対応し、列順は `source`、展開したスペクトル特徴量です。
`append_pca_scores` の戻り値も `LazyFrame` で、collect 後はこの列順の末尾に `pca-1` から
`pca-{n_component}` が追加されます。特徴量列は間引き後も全入力で同じ集合となり、波長、
`Step`、`Sequence`、`StepTime` の順で決定的に並びます。PCA の特徴量と主成分数の上限は、
この間引き後の特徴量集合から決まります。
