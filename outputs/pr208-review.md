PR #208 の全差分（22 ファイル）と関連コードを確認しました。修正をお願いしたい点は 2 件です。

1. **[P2] ストレージが使えない場合も、列の表示設定を表の更新後まで保持してください。**

   対象: [`src/flat_pca/webui/datatable/static/datatable.js:361–365`](https://github.com/ribuka/flat-pca/blob/674c76f5f64ec62989f85d29eec107c2e8928a5f/src/flat_pca/webui/datatable/static/datatable.js#L361-L365)

   `storeHiddenColumns()` は保存エラーを無視しますが、`htmx:afterSwap` のたびに `syncColumns(root, true)` がストレージから設定を読み直します。保存に失敗した場合、空の設定、または最後に保存できた古い設定でチェックボックスと CSS が上書きされます。Chrome で `Storage.prototype.setItem` に `QuotaExceededError` を送出させ、Columns で `lot` を隠してから検索すると、非表示にした列が再表示され、列数も 6 から 7 に戻ることを再現しました。README とコード内コメントが約束している「ストレージが使えないときはページを離れるまで有効」が成立しません。表ごとにメモリ上でも最新の設定を保持し、保存できない場合も検索・ソート・ページ移動の swap をまたいで使ってください。ストレージの読み書き失敗を含む E2E テストが必要です。

2. **[P2] フィルタチップでは境界値の精度を落とさないでください。**

   対象: [`src/flat_pca/webui/datatable/chips.py:41–45`](https://github.com/ribuka/flat-pca/blob/674c76f5f64ec62989f85d29eec107c2e8928a5f/src/flat_pca/webui/datatable/chips.py#L41-L45)

   `_range_label()` がセル表示用の `format_value()` を使うため、浮動小数点の境界値は有効数字 6 桁に丸められます。`min__score=1.000001&max__score=1.000002` を `parse_state()` に渡すと、実際の条件はその精度で保持される一方、チップは `1 ≤ score ≤ 1` になります。この表示からは実際に適用されている条件を確認できず、表示上の範囲と一致しない結果になります。日時でも同じ関数が小数秒を省略します。チップには実際の境界値を識別できる表記を使い、近接した浮動小数点の上下限と小数秒を含む日時をテストしてください。

検証:

- `uv run -m pytest --tb=short -q`: 1423 passed, 93 deselected。
- 関連テスト: 245 passed。
- 変更箇所に関連する Chrome E2E: 36 passed。
- `uv run -m ruff check .` と `git diff --check`: 成功。
- 上記 2 件は一時的な再現スクリプトで確認しました。列設定については通常のストレージでは非表示が保たれ、書き込み失敗時に再表示されることを Chrome で比較しています。
- [GitHub CI の E2E](https://github.com/ribuka/flat-pca/actions/runs/38021595557/job/114123576138) は 92 passed / 1 failed でした。`test_changing_the_run_clears_the_files` が実行切り替え後に `.layout` の `inert` 解除を 5 秒待って失敗しています。同じテストを含むローカル E2E は通過しており、この CI 失敗が今回の変更に起因するかは未確認です。

レビュー対象コミット: `674c76f5f64ec62989f85d29eec107c2e8928a5f`。
