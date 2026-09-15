# Ralph runner

`ralph_runner.py` は、`RALPH.md` の一般原則と `RALPH_PROJECT.md` の
リポジトリ設定に従い、Codex CLI を繰り返し起動するローカル自動実行ツールです。
各 Codex プロセスは1 taskだけを処理し、完了した場合だけ次のプロセスを起動します。

## 前提条件

- Git リポジトリのルートで実行する。
- `codex` CLI がPATH上にあり、認証済みである。
- 実行開始時のworking treeがcleanである。
- `RALPH.md`、`RALPH_PROJECT.md`、`TASKS.md`、`PROGRESS.md`、および本runnerの
  セットアップ変更を先にコミットする。

runnerは既存のユーザー変更を誤ってtaskのコミットに含めないため、dirty
worktreeでは終了コード23で停止します。

## 実行

まず、実行せずに前提条件と実行コマンドを確認します。

```powershell
uv run -m scripts.ralph_runner --dry-run
```

通常実行では、全taskが完了するか、停止条件に達するまで最大20 loopを実行します。

```powershell
uv run -m scripts.ralph_runner
```

loop数、Codex実行ファイル、モデル、プロンプトは必要に応じて指定できます。

```powershell
uv run -m scripts.ralph_runner --max-loops 8
uv run -m scripts.ralph_runner --model <model>
uv run -m scripts.ralph_runner --codex <codex-executable>
uv run -m scripts.ralph_runner --prompt-file <prompt-file>
uv run -m scripts.ralph_runner --auto-approve
```

```powershell
uv run -m scripts.ralph_runner --model "gpt-5.6-sol" --max-loops 1
```

すべてのオプションと終了コードは以下で確認できます。

```powershell
uv run -m scripts.ralph_runner --help
```

## 1 loopの流れ

1. Git rootとclean working treeを確認する。
2. `scripts/ralph_prompt.md`を渡して`codex exec`を1回実行し、`RALPH.md`、
   `RALPH_PROJECT.md`、残りのプロジェクト文書を所定の順序で読み込ませる。
3. 最終メッセージの最後の非空行を`RALPH.md`の終了トークンとして検証する。
4. Codexがcommitしていないことを確認し、runnerが当該loopの変更をステージして1コミットする。続けてGitがcleanであることと、新しいコミット数を検証する。
5. `TASK_COMPLETED: TASK-XXX`なら次loopを開始し、それ以外では停止する。

デフォルトでは`workspace-write` sandboxを使い、Codexからの確認を受けます。
無人実行が必要な場合だけ`--auto-approve`を指定してください。この場合は
`--approve-for-me`を渡しますが、競合する`--sandbox`は渡しません。runner自身は
push、pull、publish、deployを行いません。

## 停止条件と終了コード

| 最終トークンまたは状態 | runnerの動作 | 終了コード |
| --- | --- | --- |
| `TASK_COMPLETED: TASK-XXX` | runnerが1コミットして次loopへ進む | 継続 |
| `ALL_TASKS_COMPLETED` | 正常終了 | 0 |
| `TASK_INCOMPLETE: TASK-XXX` | 停止 | 20 |
| `TASK_BLOCKED: TASK-XXX` | 停止 | 21 |
| Codex実行失敗 | 停止 | 10 |
| 不正な最終トークン | 停止 | 11 |
| Git状態・コミット数が規約違反 | 停止 | 12 |
| 上限loop数に到達 | 停止 | 22 |
| 前提条件不足 | 停止 | 23 |

`TASK_INCOMPLETE`または`TASK_BLOCKED`で停止した場合も、loopで生じた変更はrunnerが
`wip(TASK-XXX): Ralph loop changes`としてコミットします。`PROGRESS.md`を読み、
人間による判断または仕様の補完後に改めて実行してください。

## ログ

各Codex実行の標準出力と標準エラーは、終了後に`logs/`へ次の名前で保存します。

```
ralph_YYYYMMDDTHHMMSS_NNN_status.log
```

- `YYYYMMDDTHHMMSS`: 実行開始時刻（ローカル時刻）
- `NNN`: `TASK-XXX`の数値部分を3桁で表したもの。taskを特定できない失敗では`000`
- `status`: `completed`、`incompleted`、`blocked`、`all-completed`、
  `codex-failure`、`protocol-error`、`git-error`

同一秒に同じtask・statusのログが既にある場合は、既存ログを上書きしないよう時刻を
1秒ずつ進めた名前を使います。

## 関連ファイル

- `ralph_runner.py`: Codex起動、終了トークン、Git状態を検証するrunner本体
- `ralph_prompt.md`: Codexへ渡す1 loop用プロンプト
- `../RALPH.md`: 1 taskの選択、loop手順、完了・blocked判定、終了トークンの正本
- `../RALPH_PROJECT.md`: 台帳、検証、fixture、runner、コミット運用の正本

実装は[Codexの非対話実行ドキュメント](https://learn.chatgpt.com/docs/non-interactive-mode)
にある`codex exec`の運用方法に基づいています。
