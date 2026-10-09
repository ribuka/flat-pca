PR #187 の全差分（仕様・JavaScript・追加 E2E テスト）と関連するフォーム生成・検証処理を確認しました。修正が必要な点が1件あります。

### [P2] 学習対象の変更後、無効な範囲設定がフォーム送信を妨げる

対象: [`src/flat_pca/webui/static/app.js:496–497`](https://github.com/ribuka/flat-pca/blob/2c47c770f59c49794ff9f9a72471351d0ec5210a/src/flat_pca/webui/static/app.js#L496-L497)（`showFitFormValues`）

数値入力へ保存値を無条件に戻すため、Data selection で学習対象を変更すると、以前の catalog 範囲が新しい入力欄の `min` / `max` と矛盾します。例えば、波長範囲が 400–402.5 nm の対象で Components だけを編集した後、401–402 nm の対象へ変更して `/fit` に戻ると、`wavelength_range_lower/upper` と `w_normalization_range_lower/upper` に 400 / 402.5 が復元されます。

両方の範囲設定が未チェックでも、その数値入力には `disabled` が付いていないため HTML の制約検証が働き、`form.checkValidity()` が `false` になります。その結果、利用者が範囲を有効にしていなくても「Run fit」を送信できません。追加の E2E 再現テストで確認し、PR の保存・復元処理を外した比較ケースでは新しい catalog の初期値が使われて検証を通りました。

無効な範囲欄をブラウザーの制約検証から外すなど、設定値の保持と学習対象変更後の送信が両立するようにしてください。catalog 範囲の異なる対象へ切り替えた場合の E2E テストも必要です。

確認結果:

- 追加の設定保持テストと既存 fit 画面の E2E テスト: 8 passed
- フォーム・ルートの関連テスト: 41 passed
- `uv run -m ruff check .`: 成功
- 一時的な追加再現テスト: 上記の問題を再現。保存・復元処理を外した比較ケース、およびナビゲーションのリンクを別タブで開くケースは成功

pytest はすべてリポジトリの指示どおりサンドボックス外で実行しました。
