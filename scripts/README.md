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
uv run -m scripts.ralph_runner --codex-timeout-sec 1800
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
2. `TASKS.md`から依存関係、priority、task IDの順に着手taskを決定する。
3. `scripts/ralph_prompt.md`を渡して`codex exec`を1回実行し、`RALPH.md`、
   `RALPH_PROJECT.md`、残りのプロジェクト文書を所定の順序で読み込ませる。
4. 最終メッセージの最後の非空行を`RALPH.md`の終了トークンとして検証し、
   事前に決定したtask IDと一致することを確認する。
5. Codexがcommitしていないことを確認し、runnerが当該loopの変更をステージして1コミットする。続けてGitがcleanであることと、新しいコミット数を検証する。
6. `TASK_COMPLETED: TASK-XXX`なら更新後の台帳を再確認する。未完了taskが残って
   いれば次loopを開始し、全taskが完了していればCodexを再起動せず正常終了する。
   それ以外の終了トークンでは停止する。

## terminalログ

runner全体の開始時に未完了task数、全task数、loop上限を表示し、終了経路に
かかわらず最後に終了を表示します。各loopではloop番号を表示し、着手可能なtaskが
決定した場合はCodex起動前にtask IDを表示します。

```text
Ralph runner start
Incompleted tasks: 3 / All tasks: 9
Total loops: 20
Ralph loop start (1)
Ralph task TASK-007 started
Ralph runner end
```

これらのstatusログはLoguruを介してterminalへ出力されます。`--dry-run`で表示する
Codexコマンド文字列だけは、実行対象そのものを確認する出力として`print()`を使います。

デフォルトでは`workspace-write` sandboxを使います。通常のリポジトリ操作はsandbox内で
行い、必要な承認はCodexの既定の承認フローに従います。pytestだけはOSの一時ディレクトリを
使うため、sandbox外実行が必要になる場合があります。個人用のCodex rulesでpytestを`allow`
している場合は、この承認を省略できます。無人実行が必要な場合だけ`--auto-approve`を指定してください。
この場合は`--approve-for-me`を渡しますが、競合する`--sandbox`は渡しません。runner自身は
push、pull、publish、deployを行いません。

各 Codex 子プロセスでは、無制限のネットワーク再接続を既定で無効化します。接続に
失敗してCodexが終了した場合や、既定で30分（1,800秒）の実行上限を超えた場合は、
子プロセスを失敗として扱い、
`--api-retry-count` の設定に従って再試行します。長時間の実装・検証を行う場合は
`--codex-timeout-sec` に秒数を指定して上限を延長できます。

## 停止条件と終了コード

| 最終トークンまたは状態 | runnerの動作 | 終了コード |
| --- | --- | --- |
| `TASK_COMPLETED: TASK-XXX`（未完了taskあり） | runnerが1コミットして次loopへ進む | 継続 |
| `TASK_COMPLETED: TASK-XXX`（全task完了） | 追加loopを起動せず正常終了 | 0 |
| `TASK_INCOMPLETE: TASK-XXX` | 停止 | 20 |
| `TASK_BLOCKED: TASK-XXX` | 停止 | 21 |
| 未完了taskあり・着手可能taskなし | Codexを起動せず停止 | 21 |
| Codex実行失敗 | 停止 | 10 |
| 不正な最終トークン | 停止 | 11 |
| Git状態・コミット数が規約違反 | 停止 | 12 |
| 上限loop数に到達 | 停止 | 22 |
| 前提条件不足 | 停止 | 23 |

`TASK_INCOMPLETE`または`TASK_BLOCKED`で停止した場合も、loopで生じた変更はrunnerが
`wip(TASK-XXX): Ralph loop changes`としてコミットします。`PROGRESS.md`を読み、
人間による判断または仕様の補完後に改めて実行してください。

## ログ

各Codex実行の標準出力と標準エラーは、開始時から`logs/`へ次の名前で書き込みます。

```
ralph_YYYYMMDDTHHMMSS_NNN_status.log
```

- `YYYYMMDDTHHMMSS`: 実行開始時刻（ローカル時刻）
- `NNN`: `TASK-XXX`の数値部分を3桁で表したもの。taskを特定できない失敗では`000`
- `status`: `completed`、`incompleted`、`blocked`、`codex-failure`、
  `protocol-error`、`git-error`

実行中は、開始時点で確定している時刻と選択taskを使った
`ralph_YYYYMMDDTHHMMSS_NNN_running.log`として生成します。終了時に同じファイルを
最終status名へ変更します。そのため、実行中のログもどのtaskのものか判別できます。

同一秒に同じtask・statusのログが既にある場合は、既存ログを上書きしないよう時刻を
1秒ずつ進めた名前を使います。

API再試行中は同じ実行中ログを上書きして使うため、試行ごとの個別ログは残りません。
すべての試行が失敗した場合は、最後の試行の出力だけを`NNN_codex-failure.log`（または
`NNN_protocol-error.log`）として保存します。

最終taskの完了によって全taskが完了した場合、そのCodex実行のログは個別taskの結果を
表す`NNN_completed.log`として保存します。完了確認だけを行う追加loopは生成しません。
起動時点ですでに全taskが完了している場合も、Codexを起動せず正常終了します。

## 関連ファイル

- `ralph_runner.py`: Codex起動、終了トークン、Git状態を検証するrunner本体
- `ralph_prompt.md`: Codexへ渡す1 loop用プロンプト
- `../RALPH.md`: 1 taskの選択、loop手順、完了・blocked判定、終了トークンの正本
- `../RALPH_PROJECT.md`: 台帳、検証、fixture、runner、コミット運用の正本

実装は[Codexの非対話実行ドキュメント](https://learn.chatgpt.com/docs/non-interactive-mode)
にある`codex exec`の運用方法に基づいています。
