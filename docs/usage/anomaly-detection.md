# PCA による異常検知（T² と Q 統計量）

fit 済みの `PcaModel` を使い、各サンプルが学習データの分布からどれだけ離れているかを PCA スコア空間のマハラノビス距離（Hotelling T²）で表します。
「この値を超えたら異常」という管理限界（UCL）も同時に得られます。

定義や検証規則の詳細は [SPEC.md](../../SPEC.md) の「マハラノビス距離仕様」を参照してください。

## 距離列を付ける

`append_pca_scores`（または `transform_pca`・`fit_and_transform_pca`）に `MahalanobisConfig` を渡すと、PCA スコア列の後ろに次の3列が追加されます。
既定値（`mahalanobis=None`）では追加されません。

| 列名 | 内容 |
| --- | --- |
| `mahalanobis_sq` | マハラノビス距離の二乗 $D^2 = \sum_{j=1}^{k} t_j^2 / \lambda_j$ |
| `mahalanobis_ucl` | 管理限界（全行で同じ値） |
| `mahalanobis_exceeds_ucl` | `mahalanobis_sq > mahalanobis_ucl` を表す bool |

```python
import polars as pl

from flat_pca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
)
from flat_pca.feature_engineering.pca import MahalanobisConfig

flattened: pl.LazyFrame = preprocess_and_flatten(paths, **preprocess_cfg)
pca = flatten_pca(flattened=flattened)

result: pl.DataFrame = append_pca_scores(
    pca,
    flattened,
    mahalanobis=MahalanobisConfig(),
).collect()

print(result.select("source", "mahalanobis_sq", "mahalanobis_exceeds_ucl"))
```

学習に使っていない新しいデータにも、同じモデルでそのまま適用できます。

```python
new_flattened = preprocess_and_flatten(new_paths, **preprocess_cfg)
new_result = append_pca_scores(
    pca,
    new_flattened,
    mahalanobis=MahalanobisConfig(),
).collect()
```

## UCL だけを取得する

グラフの水平線などに使うときは、`PcaModel.get_mahalanobis_threshold` で UCL をスカラーとして取得できます。
引数の意味と既定値は `MahalanobisConfig` と同じです。

```python
ucl: float = pca.get_mahalanobis_threshold()
ucl_05: float = pca.get_mahalanobis_threshold(alpha=0.05)
```

## 可視化

距離をサンプル順にプロットし、UCL を水平線として重ねます。

```python
import plotly.graph_objects as go

ucl = pca.get_mahalanobis_threshold()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=result["source"],
        y=result["mahalanobis_sq"],
        mode="lines+markers",
        marker={
            "color": [
                "crimson" if exceeds else "steelblue"
                for exceeds in result["mahalanobis_exceeds_ucl"]
            ]
        },
        name="Mahalanobis distance²",
    )
)
fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
fig.update_layout(xaxis_title="source", yaxis_title="D²")
fig.show()
```

## 設定の変え方

```python
config = MahalanobisConfig(
    cumulative_explained_variance=0.95,  # 使う主成分の選択
    alpha=0.05,  # UCL の有意水準（0 < alpha < 1）
    distance_column="t2",
    ucl_column="t2_ucl",
    exceeds_ucl_column="t2_alarm",
)
result = append_pca_scores(pca, flattened, mahalanobis=config).collect()
```

- `cumulative_explained_variance` は `get_feature_contribution_ranking` と同じ規則です。
  - `float`（`0 < 値 <= 1`）：累積寄与率がこの値に達するまでの主成分を使う。
  - `int`（`>= 1`）：先頭からこの本数の主成分を使う。
  - `None`：fit したすべての主成分を使う。
- `alpha` を小さくすると UCL が高くなり、異常と判定されにくくなります。
- 3つの列名は空でなく、互いに異なり、入力の列名や `pca-1` などのスコア列名と重ならない必要があります。

### 主成分数の既定値が `0.9` である理由

`flatten_pca` は既定で主成分数をサンプル数までfitします。この状態で全成分を使うと、次の問題が起きます。

- 最後の成分の分散がほぼ 0 になり、距離が発散する（この場合は `ValueError` になります）。
- 使う主成分数 $k$ が $n - 1$（$n$ は学習サンプル数）のとき、学習データの $D^2$ が全サンプルで $(n-1)^2/n$ と同じ値になり、異常を区別できない。

そのため、既定では累積寄与率 90% までの主成分だけを使います。
なお、使う主成分数が学習サンプル数以上になる設定も `ValueError` になります。

## 注意点

- UCL は「新しい観測値」向けの F 分布の式です。学習データそのものを評価すると、UCL はやや保守的になります。
- T² は、学習時の変動パターンの範囲内で振れ幅が大きいサンプルしか捉えられません。
  学習時に見たことのない形の変化（新しいピークの出現など）は、部分空間の外側の残差（Q 統計量、SPE）で捉える必要があります。
  次の「Q 統計量（SPE）による異常検知」を参照してください。

## Q 統計量（SPE）による異常検知

Q 統計量（SPE：Squared Prediction Error）は、前処理後の特徴量と、先頭 $k$ 個の主成分から再構成した値との差の二乗和です。

$$Q = \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert^2, \quad \hat{\mathbf{x}} = \boldsymbol{\mu} + \sum_{j=1}^{k} t_j \mathbf{w}_j$$

定義や検証規則の詳細は [SPEC.md](../../SPEC.md) の「Q統計量（SPE）仕様」を参照してください。

### T² と Q の使い分け

| 指標 | 捉えるもの | 例 |
| --- | --- | --- |
| T²（`mahalanobis_sq`） | 学習時の変動パターンの中で、振れ幅が大きいサンプル | 既存のピークが普段より大きく伸びる |
| Q（`spe`） | 主成分の部分空間の外側、学習時にない形の変化 | 新しいピークが出る、ベースラインの形が変わる |

多変量統計的プロセス管理では、T² と Q を対にして使うのが一般的です。
スペクトルの異常検知では、T² よりも Q のほうが異常に敏感な場合もあります。

### Q 列を付ける

`append_pca_scores`（または `transform_pca`・`fit_and_transform_pca`）に `SpeConfig` を渡すと、PCA スコア列の後ろに次の3列が追加されます。
`MahalanobisConfig` も渡した場合は、マハラノビス距離の3列の後ろに追加されます。
既定値（`spe=None`）では追加されません。

| 列名 | 内容 |
| --- | --- |
| `spe` | Q 統計量 |
| `spe_ucl` | 管理限界（全行で同じ値） |
| `spe_exceeds_ucl` | `spe > spe_ucl` を表す bool |

```python
from flat_pca.feature_engineering.pca import MahalanobisConfig, SpeConfig

result: pl.DataFrame = append_pca_scores(
    pca,
    flattened,
    mahalanobis=MahalanobisConfig(),
    spe=SpeConfig(),
).collect()

print(
    result.select(
        "source",
        "mahalanobis_sq",
        "mahalanobis_exceeds_ucl",
        "spe",
        "spe_exceeds_ucl",
    )
)
```

`SpeConfig` の引数は `MahalanobisConfig` と同じ意味です（列名の引数は `spe_column`・`ucl_column`・`exceeds_ucl_column`）。
主成分数の既定値も同じ `0.9` にそろえてあり、T² と Q が同じ部分空間の内側と外側を見るようになっています。
片方だけ変える場合は、両方の `cumulative_explained_variance` をそろえてください。

### UCL だけを取得する

```python
spe_ucl: float = pca.get_spe_threshold()
spe_ucl_05: float = pca.get_spe_threshold(alpha=0.05)
```

UCL は Jackson–Mudholkar の近似式で、残差が正規分布に従うことを前提にしています。

### T² と Q を並べて可視化する

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("T²", "Q (SPE)"))
for row, (value_column, exceeds_column, ucl) in enumerate(
    [
        ("mahalanobis_sq", "mahalanobis_exceeds_ucl", pca.get_mahalanobis_threshold()),
        ("spe", "spe_exceeds_ucl", pca.get_spe_threshold()),
    ],
    start=1,
):
    fig.add_trace(
        go.Scatter(
            x=result["source"],
            y=result[value_column],
            mode="lines+markers",
            marker={
                "color": [
                    "crimson" if exceeds else "steelblue"
                    for exceeds in result[exceeds_column]
                ]
            },
            name=value_column,
        ),
        row=row,
        col=1,
    )
    fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL", row=row, col=1)
fig.update_xaxes(title_text="source", row=2, col=1)
fig.show()
```

T² と Q を横軸・縦軸にとった散布図にすると、どちらの指標で外れたかを一目で区別できます。

```python
fig = go.Figure(
    go.Scatter(
        x=result["mahalanobis_sq"],
        y=result["spe"],
        mode="markers",
        text=result["source"],
    )
)
fig.add_vline(x=pca.get_mahalanobis_threshold(), line_dash="dash")
fig.add_hline(y=pca.get_spe_threshold(), line_dash="dash")
fig.update_layout(xaxis_title="T²", yaxis_title="Q (SPE)")
fig.show()
```

### 注意点

- 使う主成分数が階数の上限（学習サンプル数と特徴量数の小さいほう）に達し、残差側に分散が残らない設定は `ValueError` になります。
  `SpeConfig(cumulative_explained_variance=None)` は、fit したすべての成分を使うため、多くの場合この条件に当たります。
- `max_n_component` などで成分数を打ち切って fit したモデルでは、fit していない成分の分散をその平均値（`pca.noise_variance_`）で近似するため、UCL の精度が落ちます。
  `flatten_pca` は既定で階数の上限まで fit するため影響はありません。
- 残差側で1つの成分の分散が大きく、残りの多数の成分の分散が小さい場合は、Jackson–Mudholkar の式が使えないため、その元になった Box の近似（尺度を掛けた χ² 分布）で UCL を求めます。
- 実データで誤報が多い場合は、学習データの Q の分布（`append_pca_scores(pca, flattened, spe=SpeConfig())` の `spe` 列の分位点など）から経験的に閾値を決めることも検討してください。
