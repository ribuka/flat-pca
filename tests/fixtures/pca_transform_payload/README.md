# PCA 変換状態 JSON フィクスチャ

`PcaModel.to_transform_json` が書き出す現行形式（平たいスキーマ）の JSON を、保存済みモデルの互換性テスト用に固定したものです。
前処理の内部構造を変えても、これらの JSON を読み込んで同じ変換結果を再現できることを確かめます。

各ファイルは次の3つのキーを持ちます。

- `input`: `transform_pca` に渡す入力フレーム（列名 → 値のリスト。欠損は `null`）
- `payload`: `to_transform_json` が書き出した変換状態
- `expected`: フィクスチャ作成時の `transform_pca` の出力

| ファイル | 補完 | 外れ値 | スケーリング |
| --- | --- | --- | --- |
| `median_winsorize_robust.json` | `median` | `winsorize` | `robust` |
| `kmeans_drop_pareto.json` | `kmeans`（3クラスタ） | `drop`（IQR 倍率 2.0） | `pareto` |
| `drop_none_none.json` | `drop` | なし | `none`（NumPy 高速経路） |
| `legacy_median_zscore.json` | `median` | `"none"` | `z-score` |

`legacy_median_zscore.json` は、kmeans 補完と外れ値の閾値が追加される前の形式です。
`impute_kmeans_*`、`iqr_multiplier`、`*_bounds` の各キーを持たず、外れ値処理なしを `"none"` で表します。
