# Ralph loop作業規約

この文書は、1回のloopで`TASKS.md`に定義されたタスクを1件だけ処理するための作業規約である。

## 参照するファイル

各loopの開始時に、次の順でファイルを読む。

1. `AGENTS.md`
2. `SPEC.md`
3. `TASKS.md`
4. `PROGRESS.md`（存在する場合）
5. 選択したタスクに関係する既存コードとテスト

ファイルに書かれていない要件を推測で追加しない。仕様間に矛盾がある場合は、実装せずタスクをblockedとして報告する。

## タスクの選択

1. `Status`が`pending`のタスクだけを候補にする。
2. `depends_on`に記載されたすべてのタスクが`completed`であることを確認する。
3. 候補のうち`priority`が最も小さいタスクを選ぶ。
4. 同じpriorityの候補が複数ある場合は、`id`の昇順で選ぶ。
5. 1回のloopで選択、実装、完了できるタスクは1件だけとする。

未完了タスクがすべて依存関係で停止している場合は、ファイルを変更せずblockedとして終了する。

## 1回のloop

1. 選択したタスクの`requirements`、`tests`、`acceptance_commands`を読む。
2. コードベースを検索し、要件がすでに実装されていないか確認する。
3. タスクの要件を検証するテストを追加または更新する。
   - Flatten-PCAに関するタスクでは、`AGENTS.md`の規約に従い`tests/fixtures/real_subset`を読み込むテストを少なくとも1件含める。
4. 対象テストを実行し、未実装の要件に対して適切に失敗することを確認する。
5. 選択したタスクだけを満たす最小限の実装を行う。
6. 対象テストを再実行する。
7. タスクに定義されたすべての`acceptance_commands`を実行する。
8. `requirements`を1項目ずつ実装およびテストと照合する。
9. 完了条件をすべて満たした場合だけ、選択したタスクの`Status`を`completed`に変更する。
10. `PROGRESS.md`にloopの結果を英語で追記する。
11. このloopで変更したファイルだけを1件のGit commitにまとめて終了する。

テストだけを通すために要件を弱めたり、テストを削除、skip、xfailしたりしてはならない。

## 完了判定

選択したタスクは、次の条件をすべて満たした場合だけ完了とする。

- `requirements`の全項目を満たしている。
- `tests`に対応する自動テストが存在する。
- 追加または変更したテストが成功する。
- 既存テストを含む`uv run -m pytest`が成功する。
- すべての`acceptance_commands`が終了コード0で完了する。
- 関係のない機能や公開APIを変更していない。
- 一時的なデバッグコード、生成物、未使用ファイルが残っていない。

テストが成功しても、要件との照合に失敗した場合は完了ではない。

## 失敗およびblocked

- 完了条件を満たせない場合、タスクの`Status`を`completed`にしない。
- 人間の判断、資格情報、外部サービス、未定義の仕様が必要な場合は、それ以上推測で進めない。
- `PROGRESS.md`に、実行した内容、成功した検証、失敗内容、次に必要な判断を記録する。
- 選択したタスクとは別の問題を発見した場合、現在のタスクに必要でなければ修正しない。
- 新しい作業が必要な場合は、`TASKS.md`を無断で拡張せず、`PROGRESS.md`にタスク候補として記録する。

## 状態の更新

`TASKS.md`では、原則として選択したタスクの`Status`だけを変更する。要件、テスト条件、依存関係および優先度をloop中に変更してはならない。

`PROGRESS.md`には次の形式で追記する。

見出し、結果、変更内容、テスト結果、要件確認、注記を含む追記内容はすべて英語で記載する。

```markdown
## YYYY-MM-DD HH:MM - TASK-XXX

- Result: completed | pending | blocked
- Changes: Implementation or documentation changes made in this loop
- Tests: Commands run and their results
- Requirements: Verification result for each requirement
- Notes: Information required by the next loop
```

## loopの終了出力

- 1件のタスクを完了した場合: `TASK_COMPLETED: TASK-XXX`
- タスクを完了できなかった場合: `TASK_INCOMPLETE: TASK-XXX`
- 人間の判断などで停止した場合: `TASK_BLOCKED: TASK-XXX`
- すべてのタスクが完了済みの場合: `ALL_TASKS_COMPLETED`

終了出力は状態を偽ってはならない。1回のloopが終了したら、別のタスクを開始しない。

## Gitおよび外部操作

- リポジトリに変更がある各loopは、`TASKS.md`と`PROGRESS.md`の状態更新を含め、このloopで変更したファイルだけを終了時に1件のcommitへまとめる。
- commit前に差分とstagedファイルを確認し、loop開始前から存在するユーザーの変更や、選択したタスクと無関係な変更をcommitへ含めない。
- タスク完了時のcommit messageには`TASK-XXX`と変更内容を含める。
- タスク未完了またはblocked時に作業内容を保存する必要がある場合は、`wip(TASK-XXX): ...`としてcommitし、失敗した検証と残作業を`PROGRESS.md`へ英語で記録する。
- ファイル変更がない終了確認だけのloopでは、空commitを作成しない。
- loopはpush、pull、公開、デプロイを行わない。
- 破壊的な操作やリポジトリ外への書き込みを行わない。
- ユーザーが作成した未コミット変更を上書き、削除、revertしない。
