# 寄与率の表と特徴量の再構成

fit 済みの `PcaModel` から、成分ごとの寄与率・累積寄与率の表を取得して成分数を決める方法と、PCA スコアから元のスケールの特徴量（スペクトル）を再構成する方法を示します。

定義や検証規則の詳細は [SPEC.md](../../SPEC.md) の「寄与率の表と特徴量の再構成仕様」を参照してください。

## 寄与率の表

`PcaModel.get_explained_variance_table` は、fit したすべての成分について 1 成分 1 行の表を返します。

| 列名 | 内容 |
| --- | --- |
| `component` | 1 始まりの成分番号 |
| `pca_column` | スコア列名（`pca-1` など） |
| `explained_variance` | 成分の分散 |
| `explained_variance_ratio` | 寄与率 |
| `cumulative_explained_variance` | 累積寄与率 |

```python
import polars as pl

from flat_pca.feature_engineering import (
    append_pca_scores,
    flatten_pca,
    preprocess_and_flatten,
)

flattened: pl.LazyFrame = preprocess_and_flatten(paths, **preprocess_cfg)
pca = flatten_pca(flattened=flattened)

table: pl.DataFrame = pca.get_explained_variance_table()
print(table.head(10))
```

累積寄与率が 90% に届く成分番号は、表の最初の該当行で確認できます。
この番号は、`cumulative_explained_variance=0.9` を受け取る各メソッド（`get_feature_contribution_ranking`、`MahalanobisConfig`、`reconstruct` など）が選ぶ成分数と一致します。
成分数を打ち切って fit したモデルでは最後の成分でも 90% に届かないことがあり、その場合に各メソッドが選ぶのは fit したすべての成分（表の最後の行）です。

```python
reaching = table.filter(pl.col("cumulative_explained_variance") >= 0.9)
k_90 = reaching["component"][0] if reaching.height > 0 else table["component"][-1]
```

### スクリープロット

寄与率を棒、累積寄与率を折れ線で重ねて、成分数を決める目安にします。

```python
import plotly.graph_objects as go

head = table.head(20)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=head["component"],
        y=head["explained_variance_ratio"],
        name="寄与率",
    )
)
fig.add_trace(
    go.Scatter(
        x=head["component"],
        y=head["cumulative_explained_variance"],
        mode="lines+markers",
        name="累積寄与率",
    )
)
fig.add_hline(y=0.9, line_dash="dash", line_color="gray")
fig.update_layout(xaxis_title="component", yaxis_title="ratio")
fig.show()
```

## 特徴量の再構成

`PcaModel.reconstruct` は、PCA スコアから**スケーリング前の元のスケール**の特徴量を再構成します。
`cumulative_explained_variance` で使う成分を先頭 $k$ 成分に絞ると、ノイズを除いた近似スペクトルが得られます（既定の `None` は fit したすべての成分を使います）。

```python
scores: pl.DataFrame = append_pca_scores(pca, flattened).collect()

# 先頭 3 成分だけで再構成する
denoised: pl.DataFrame = pca.reconstruct(scores, cumulative_explained_variance=3)

# 累積寄与率 90% までの成分で再構成する
denoised_90: pl.DataFrame = pca.reconstruct(scores, cumulative_explained_variance=0.9)
```

- 出力は、`scores` のスコア列・特徴量列以外の列（`source` など）を先頭に残し、その後ろに再構成した特徴量列を `pca.columns` の順に並べたものです。
  `append_pca_scores` の出力のように特徴量列を含む場合、その列は再構成値に置き換わります。
- 使うスコアに null / NaN を含む行は、再構成値がすべて NaN になります（行は削除されません）。
- 戻せるのはスケーリングだけです。
  欠損値補完や外れ値処理（winsorize など）を使ったモデルでは、再構成値は補完・クリップ後のデータの近似になります。
  `flatten_pca` の既定（補完なし・外れ値処理なし・スケーリングなし）では影響はありません。

### スコアを動かしてスペクトル上の意味を見る

スコア列を任意の値で作って渡すこともできます。
たとえば第 1 成分のスコアだけを変えたときの特徴量の変化から、その成分がスペクトル上で何を表すかを確かめられます。

```python
pc1 = scores["pca-1"]
grid = pl.DataFrame(
    {
        "pca-1": [pc1.min(), 0.0, pc1.max()],
        "pca-2": [0.0, 0.0, 0.0],
    }
)
along_pc1: pl.DataFrame = pca.reconstruct(grid, cumulative_explained_variance=2)
```

### メモリ使用量

flatten 特徴量は数千〜数万列になることがあるため、再構成結果は行数に比例して大きくなります。
入力は `pl.DataFrame` に限られるので、必要な行だけを絞り込んでから渡してください。

```python
target = scores.filter(pl.col("source") == "sample_001.parquet")
reconstructed = pca.reconstruct(target, cumulative_explained_variance=0.9)
```
