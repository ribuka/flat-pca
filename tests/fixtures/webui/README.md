# Web UI テスト用フィクスチャ

`tests/webui/` のテストが `tmp_path` へコピーして使います。相対パスは `settings.toml` のあるディレクトリ基準で解決されます。

- `data/run-1.parquet`、`data/run-2.parquet`、`data/sub/run-10.parquet`：各 7 行。`(Step, Sequence)` は `(1, 1)`・`(2, 1)`・`(2, 2)`、波長列は `400.0nm`・`401.0nm`・`402.5nm`。
- `meta.csv`：`run-1`・`run-2` と、ファイルが存在しない `ghost` の 3 行。`run-10` の行は無い。`note` 列は settings.toml に無いため取り込まれない。
- `settings.toml`：`workspace.dir` は `workspace`（コピー先に作られる）。
