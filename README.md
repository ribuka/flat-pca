# SPCA

SPCA provides a deterministic Flatten-PCA workflow for time-series spectral
data stored in Parquet files.

## 使い方

`flatten_pca` は、同じ測定条件（`Time`、`Step`、`Sequence`、波長列）を持つ
複数の Parquet ファイルを、1 ファイルにつき 1 行の特徴量へ展開し、PCA スコアを
付加します。ファイル名（拡張子を除く）がサンプル名になります。

まず、パッケージをインストールします。

```bash
uv sync
```

次の例では、`data/spectra` にある Parquet ファイルすべてを入力にし、第 1・第 2
主成分を計算します。

```python
from pathlib import Path

from spca.feature_engineering import flatten_pca

input_dir = Path("data/spectra")
paths = sorted(input_dir.glob("*.parquet"))

result = flatten_pca(paths)

# サンプル名と PCA スコアだけを確認する
print(result.select("filename", "pca-1", "pca-2"))

# 後続の解析用に保存する場合
result.write_parquet("data/flatten_pca_result.parquet")
```

`paths` には、拡張子を除いたファイル名が重複しない 1 個以上のパスを渡します。
`n_component` を省略すると、「入力ファイル数」と「展開後のスペクトル特徴量数」の小さい方まで
すべての主成分を計算します。必要な主成分数だけに絞る場合は、1 以上かつこの上限以下の整数を
指定します。入力パスの順序は結果に影響しません。

### 前処理を指定する例

必要に応じて、時間・波長方向の平滑化と規格化を同時に指定できます。窓幅は半窓幅で、
範囲の両端を含みます。単位はそれぞれ入力の `Time` と波長列名の数値部分に合わせます。

```python
result = flatten_pca(
    paths,
    n_component=3,
    t_smoothing_window=300.0,
    w_smoothing_window=5.0,
    t_normalization_range=(0.0, 9_690.0),
    w_normalization_range=(350.0, 450.0),
    t_downsampling_stride=2,
    w_downsampling_stride=3,
)
```

`t_downsampling_stride` と `w_downsampling_stride` は、全入力で共通する昇順の Time 値・
波長値について先頭から何個おきに残すかを、1 以上の整数で指定します。既定値 `1` は
間引きを行いません。

前処理は常に、時間方向平滑化、波長方向平滑化、時間方向規格化、波長方向規格化、
時間方向間引き、波長方向間引き、flatten の順で適用されます。平滑化と規格化は間引き前の
全観測値を使用します。不要な平滑化・規格化は引数を省略するか `None` を渡してください。

## 入力 Parquet の形式

各 Parquet ファイルには数値型の `Time`、`Step`、`Sequence` 列が必要です。その他の列は
スペクトル値の数値列とし、列名は `350.0nm` のような小数第 1 位までの波長形式にします。
値は欠損・NaN・無限大を含めず、各ファイル内の `(Time, Step, Sequence)` は一意である必要が
あります。また、すべての入力ファイルでメタデータの組み合わせと波長列の集合を一致させます。

## 出力

戻り値は Polars の `DataFrame` です。1 行が入力ファイル 1 個に対応し、列順は
`filename`、展開したスペクトル特徴量、`pca-1` から `pca-{n_component}` です。特徴量列は
間引き後も全入力で同じ集合となり、波長、`Step`、`Sequence`、`Time` の順で決定的に
並びます。PCA の特徴量と主成分数の上限は、この間引き後の特徴量集合から決まります。
