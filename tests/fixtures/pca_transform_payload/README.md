# PCA 変換状態 JSON フィクスチャ

`PcaModel.to_transform_json` が書き出す JSON を、保存済みモデルの往復テスト用に固定したものです。
前処理の内部構造を変えても、これらの JSON を読み込んで同じ変換結果を再現できることを確かめます。

各ファイルは次の3つのキーを持ちます。

- `input`: `transform_pca` に渡す入力フレーム（列名 → 値のリスト。欠損は `null`）
- `payload`: `to_transform_json` が書き出した変換状態
- `expected`: フィクスチャ作成時の `transform_pca` の出力

`payload` は、前処理の各段階の状態を `impute_model`・`outlier_model`・`scaling_model` の各キーに入れ子で持ちます。
各キーの値は、対応する段階の `to_payload()` の出力です。

| ファイル | 補完 | 外れ値 | スケーリング |
| --- | --- | --- | --- |
| `median_winsorize_robust.json` | `median` | `winsorize` | `robust` |
| `kmeans_drop_pareto.json` | `kmeans`（3クラスタ） | `drop`（IQR 倍率 2.0） | `pareto` |
| `drop_none_none.json` | `drop` | なし | `none`（NumPy 高速経路） |
