PR の差分全体（41 ファイル、削除・移動されたコード、文書、テストを含む）を確認しました。汎用部品として利用する場合に、以下の 3 点は修正が必要です。

1. **[P2] 行キーの文字列化を全選択と行チェックで統一してください**  
   対象: [`macros.html:101–104`](https://github.com/ribuka/flat-pca/blob/41a2f1134cb18ddc40c92188cc836cf550ca7afe/src/flat_pca/webui/datatable/templates/datatable/macros.html#L101-L104)、[`query.py:164–166`](https://github.com/ribuka/flat-pca/blob/41a2f1134cb18ddc40c92188cc836cf550ca7afe/src/flat_pca/webui/datatable/query.py#L164-L166)。

   `matching_keys` は Polars の `cast(pl.String)`、行チェックの `value` は Jinja の `string` で生成されるため、キーの型によって値が一致しません。日時キー `datetime(2026, 1, 1)` では前者が `2026-01-01 00:00:00.000000`、後者が `2026-01-01 00:00:00`、真偽値では `true` と `True` になります。ヘッダの全選択を押しても `syncChecks()` の `keys.has(box.value)` が成立せず行がチェックされず、個別選択もヘッダの選択状態に反映されません。行チェック・全選択・初期選択に同じキー表現を使い、文字列以外のキーでも選択が成立するようにしてください。

2. **[P2] 整数のフィルタ境界を float に丸めないでください**  
   対象: [`state.py:99–105`](https://github.com/ribuka/flat-pca/blob/41a2f1134cb18ddc40c92188cc836cf550ca7afe/src/flat_pca/webui/datatable/state.py#L99-L105)。

   汎用部品は整数列も対象としますが、数値境界を必ず `float(text)` に変換するため、Int64 / UInt64 の大きな値を正しく絞り込めません。Int64 列に `9007199254740992` と `9007199254740993` を入れ、下限を `9007199254740993` にすると、解析後の下限が `9007199254740992.0` となり、両方の行が返ることを確認しました。整数リテラルの精度を保持し、Polars の比較でも列と境界が float に丸められないようにしてください。

3. **[P2] LazyFrame はページの切り出し後に収集してください**  
   対象: [`query.py:158–163`](https://github.com/ribuka/flat-pca/blob/41a2f1134cb18ddc40c92188cc836cf550ca7afe/src/flat_pca/webui/datatable/query.py#L158-L163)。

   現在はフィルタ・ソート後に全行・全列を `collect()` し、その後でページを切り出しています。`selectable=False` でも同じなので、大きな Parquet の LazyFrame を渡すと、1000 行だけ表示するリクエストでも一致する全データをメモリへ読み込みます。ページングでメモリ使用量を抑えられず、表を流用する際に応答遅延やメモリ不足を起こします。件数は集約で求め、表示行は LazyFrame 上で `slice()` してから収集し、全選択用のキーが必要な場合もキー列だけを別途収集してください。

検証: 関連テスト `tests/webui/datatable`、`test_file_table.py`、`test_catalog_query.py`、`test_routes.py`、`test_transform_routes.py` は **111 passed**。`uv run -m ruff check .` と差分の whitespace チェックも通過しました。上記 3 点は追加の再現スクリプトで確認しました。今回のレビューでは全テストスイートとブラウザ E2E は実行していません。
