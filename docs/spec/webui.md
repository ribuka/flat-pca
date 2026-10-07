# Web UI仕様（草案）

## スコープ

- ローカルPC上の1ユーザーが、Parquetファイル群の選択 → 前処理・Flatten-PCA → 可視化をブラウザから行うWeb UIを提供する。
- 構成はFastAPI + Jinja2 + htmxとし、グラフ描画はPlotly（ブラウザ側はPlotly.js）を使う。
- 将来、共有サーバー上で「同時実行1ユーザー」の運用へ移行できる構造にする（状態の集約・ジョブの単一実行）。認証・複数ユーザーの同時利用はv1の対象外とする。
- 対象外（将来検討）：
  - ブラウザからのファイルアップロード（Parquetのlazy読み込みにはランダムアクセス可能なストレージが必要なため、サーバーから見えるパスまたはURIを指定する方式とする）
  - `Time`で区間をつなげたヒートマップ（v1は`(Step, Sequence)`ごとの`StepTime`のみ）
  - 学習済みモデルによる未知データの判定

## 構成

### 配置

```
src/flat_pca/
  feature_engineering/          データ生成（既存）。UIで必要な計算関数もここに追加する
  visualize/                    データ → go.Figure の純粋関数（Notebookからも使う）
  webui/
    __main__.py                 uv run -m flat_pca.webui --settings <path> で起動
    app.py                      FastAPIアプリ生成、router登録、static mount
    settings.py                 settings.tomlの読み込み・検証
    workspace.py                Workspace（DB接続・キャッシュ・ジョブ実行器）
    routes/                     画面・機能単位のAPIRouter（薄く保つ）
    services/                   routesから呼ぶ処理本体（catalog, runs, views）
    jobs/                       サブプロセスで実行するジョブ関数
    templates/{pages,partials}/ フルページとhtmxが差し替える断片
    static/                     vendor/htmx.min.js, app.js, app.css（plotly.min.jsはplotlyパッケージ同梱版を/static/vendor/で配信）
tests/webui/
```

- UI固有の依存（`fastapi`、`uvicorn`、`jinja2`、`duckdb`、`pydantic`）はoptional dependency `webui`とし、ライブラリ本体の依存に加えない。
- htmxは`static/vendor/`に同梱する。Plotly.jsはPythonの`plotly`パッケージに同梱された`plotly.min.js`を`/static/vendor/plotly.min.js`で配信し、Python側とバージョンを揃える。いずれもCDNに依存しない。
- フォーム送信の解析に使う`python-multipart`も`webui`に含める。

### レイヤーの責務

| レイヤー | 責務 | 禁止事項 |
| --- | --- | --- |
| `feature_engineering/` | 部分スコア・単一成分再構成など数値計算 | UI・描画への依存 |
| `visualize/` | polars/NumPy → `go.Figure` | ファイルI/O、UI状態への依存 |
| `webui/services/` | 対象データの選択、ビニング、キャッシュ、成果物の読み書き | 図の見た目の決定 |
| `webui/routes/` | リクエスト検証、service呼び出し、テンプレート描画 | 計算ロジック |
| `webui/static/app.js` | `Plotly.newPlot`、クリック・スライダー → htmx/fetchの橋渡し | データ加工 |

- 図はサーバーで`figure.to_json()`し、テンプレートに埋め込んでブラウザで描画する。UI専用の描画コードは書かず、Notebookと同じ図を使う。

## 設定（settings.toml）

起動時に`--settings`で指定する。`tomllib`で読み込み、pydanticで検証する。不正な場合は起動時にエラーとする。

```toml
[workspace]
dir = "D:/flatpca-workspace"          # DuckDBファイルとruns/の置き場所

[data]
root = "D:/spectra"                   # Parquetを走査するディレクトリ
glob = "**/*.parquet"

[metadata]                            # 省略可
csv = "D:/spectra/meta.csv"
key = "file_name"                     # CSV側の結合キー列。ファイルのPath.stemと照合する

[metadata.columns]                    # 取り込む列と型。ここに無い列は無視する
lot       = { type = "category" }
date      = { type = "datetime", format = "%Y-%m-%d %H:%M:%S" }
yield_pct = { type = "number" }

[ui]
default_color_by = "lot"              # 散布図・軌跡の色分け既定
default_order_by = "date"             # T²/Q管理図の横軸順の既定（省略時はファイル名の自然順）
heatmap_max_cells = 1_200_000         # ヒートマップ送信時のセル数上限

[jobs]
artifact_dtype = "float32"            # "float32" | "float64"
memory_warn_gb = 16                   # 実行前見積もりがこれを超えたら確認を求める
```

- 型`category`・`number`・`datetime`のみを受け付ける。`format`は`datetime`のみに指定でき、省略時は形式を推定する。
- メタデータ列名に、結合キー列名と、ファイル一覧の列名（`stem`・`path`・`n_rows`・`n_steps`・`n_segments`）は使えない。
- `workspace.dir`・`data.root`・`metadata.csv`の相対パスは、settings.tomlのあるディレクトリ基準で解決する。未知のキーはエラーとする。
- メタデータの結合キーは`Path.stem`とする。`data.root`配下でstemが重複する場合はcatalog構築をエラーとする（UI経由の実行では`stem_uniqueness="error"`を使う）。
- メタデータCSVの結合キーが空または重複する場合、取り込む列が無い場合、値を型に変換できない場合はcatalog構築をエラーとする。
- CSVに無いファイル、ファイルが存在しないCSV行は件数と一覧をcatalog画面に警告表示し、処理は継続する。

## Workspace と DB

- 状態は1つの`Workspace`オブジェクト（DuckDB接続、表示用キャッシュ、ジョブ実行器）に集約し、グローバル変数に分散させない。
- DuckDBファイルは`{workspace.dir}/flatpca.duckdb`とする。書き込みは親プロセスのみが行う。
- スキーマに版管理・マイグレーションは持たない。スキーマを変えたときはWorkspaceを作り直す。`file_metadata`だけは、settings.tomlの列定義（列名・順序・SQL型）と食い違う場合に起動時に空で作り直す（次のcatalog更新で埋まる）。
- テーブル：
  - `files`：`stem`、`path`、`size`、`mtime_ns`、波長数・最小・最大、行数
  - `segments`：`stem`、`Step`、`Sequence`、行数、StepTime最大値
  - `file_metadata`：`stem` + settings.tomlで定義した列
  - `runs`：`run_id`、ジョブ種別（`catalog`等）、作成日時、状態（`queued`/`running`/`succeeded`/`failed`/`cancelled`）、設定JSON、対象ファイル数、特徴量数、成分数、所要時間、エラーメッセージ、成果物ディレクトリ
- catalogは`path`・`mtime`・`size`のいずれかが変化したファイルのみを再走査する差分更新とする。走査は`Time`・`Step`・`Sequence`列とスキーマのみを読む。

## ジョブ実行

- ピークメモリが大きくなり得る処理はサブプロセスで実行する。対象は次のとおり。
  - catalogの構築・更新
  - 前処理・flatten・PCA fit・スコア付与（1つのfitジョブ）
- 実行器はアプリ側で保持するFIFOキューと、ジョブごとに起動する子プロセス（`multiprocessing.get_context("spawn").Process`）からなる。ジョブ終了ごとに子プロセスが終了し、メモリを確実に解放する。同時に実行するジョブは1つとし、後続はキューで待つ。
  - `ProcessPoolExecutor`は使わない。ワーカーを強制終了するとプール全体が`BrokenProcessPool`となり、待機中のジョブと以後のsubmitがすべて失敗するため、キャンセルと両立しない。
  - 親プロセスの監視スレッド1本が、子の終了（`join`）を確認してから、キューの次のジョブを起動する。
  - 子が異常終了した場合（終了コードが0以外、または成果物の検証に失敗した場合）はrunを`failed`とし、キューの次のジョブへ進む。
- 子プロセスとの受け渡しはファイル経由とする。子は成果物・`progress.json`・`log.txt`をrunディレクトリに書き、大きな配列をpickleで返さない。親は完了後に成果物を検証してDBへ登録する。
- 進捗は`progress.json`（段階名・完了数・総数）とし、画面は`hx-trigger="every 1s"`でポーリングする。
- キャンセル：
  - 実行中のジョブ：子プロセスを`terminate()`し、`join()`で終了を確認してからrunを`cancelled`にする。書きかけの成果物は削除する。その後、キューの次のジョブを起動する。
  - 待機中のジョブ：キューから取り除き、runを`cancelled`にする。子プロセスは起動しない。
  - キャンセルは他のジョブの状態と、以後のジョブの投入に影響しない。
- アプリ起動時に`running`または`queued`のまま残っているrunは、前回のプロセスが異常終了したものとして`failed`にする。
- ジョブ本体は通常の関数（例：`jobs/fit_run.py::run_fit(config, run_dir)`）とし、サブプロセスを介さず単体テストできるようにする。

## run 成果物

1回の「前処理 → PCA fit」の結果一式を`{workspace.dir}/runs/{run_id}/`に保存する。

| ファイル | 内容 | 形式 |
| --- | --- | --- |
| `config.json` | 前処理・PCA・T²/Q設定、対象ファイル一覧 | JSON |
| `features.parquet` | 特徴量座標`(wavelength, Step, Sequence, StepTime)`、列順は`X.npy`の列順 | Parquet |
| `samples.parquet` | `source`、`stem` + ファイルメタデータ、行順は`X.npy`の行順 | Parquet |
| `X.npy` | 前処理・flatten・sparse列除去後、補完前の行列（欠損はNaN） | `.npy` |
| `components.npy` | `pca.components_`（`n_comp × n_features`） | `.npy` |
| `pca_state.npz` | mean・scale・explained_variance等、補完値・外れ値境界（特徴量順の配列） | `.npz` |
| `scores.parquet` | `source`、`pca-1..k`、T²・Q列 | Parquet |
| `progress.json`・`log.txt` | 進捗・ログ | JSON・テキスト |

- 巨大な数値行列は`.npy`とし、親プロセスは`np.load(..., mmap_mode="r")`で開いて必要な行だけを読む。表形式のデータはParquetとする。
- `X.npy`・`components.npy`は`jobs.artifact_dtype`で保存する（既定`float32`）。読み込み後の計算（スケーリング、スコア、再構成、残差、Q、累積和）はすべて`float64`へ変換してから行う。
- `.npy`/`.npz`から`PcaModel`を復元する関数を`pca/serialization.py`に追加する。既存のJSONシリアライズは変更しない。

## 画面

### 共通レイアウト

- 左にサイドバーを置き、各画面へのナビゲーション（未実装の画面は無効表示）と、catalogの状態・選択中のファイル数を表示する。
- サイドバーの状態は、catalog更新の開始・完了とファイル集合の選択（`catalog-started`・`catalog-updated`・`selection-updated`イベント）で再取得し、catalog実行中は2秒ごとにポーリングする。
- フォーム内でhtmxの要求を出す要素は、フォームの`hx-target`を継承しないよう`hx-target`を明示する。

### 1. データ選択

- catalogのファイル一覧を表示する（ファイル名、メタデータ列、Step/Sequence数）。メタデータ列での絞り込み・ソートができる。
- 「catalog更新」でcatalogジョブを起動する。
- 選択したファイル集合を次の画面へ渡す。

### 2. 前処理・PCA

- `PreprocessConfig`の各フィールドに対応するフォームを持つ。
  - `target_steps`はcatalogのStep一覧から複数選択する。
  - `wavelength_range`・`*_normalization_range`の初期値と範囲はcatalogから決める。
- PCA設定：`n_component`の既定は「累積寄与率0.99に達する成分数、上限1000」とし、UIで変更できる。`impute_strategy`、`MahalanobisConfig`・`SpeConfig`の`cumulative_explained_variance`と`alpha`を設定できる。
  - 成分数を打ち切るとQのUCL（$\theta_2$・$\theta_3$）の精度が下がる（SPEC.md「Q統計量（SPE）仕様」）。画面に注記する。
- 実行前に、catalogから特徴量数と必要メモリを見積もって表示する。`jobs.memory_warn_gb`を超える場合は確認を求める。
- 送信は`hx-post`とし、`PreprocessConfig`等の`ValueError`は該当フィールドの横にpartialで表示する。
- 実行中は進捗とキャンセルボタンを表示する。過去のrun一覧から再訪できる。

### 3. スペクトル探索

- 表示対象：
  - 元データ（Parquetから読む）
  - 前処理済み（`X.npy`の1行をreshape）
  - PCA成分k（`components.npy`の1行をreshape）
  - 第k成分のみの寄与（元のスケール）
  - 先頭1..k成分の累積再構成（`PcaModel.reconstruct`）
  - 残差（補完・外れ値処理後の前処理済み − 累積再構成、元のスケール）
  - Q寄与（特徴量ごとの残差の二乗。定義は「計算仕様」を参照）
- 「前処理済み」は`X.npy`の値をそのまま表示し、欠損はNaN（空白セル）として示す。再構成・残差・Q寄与は、fit時と同じ補完・外れ値処理を適用した値から計算するため欠損を含まない。
- 選択項目：ファイル（成分表示時は不要）、`(Step, Sequence)`、成分番号k。
- ヒートマップ：横軸波長、縦軸`StepTime`（0を下）。
  - セル数が`ui.heatmap_max_cells`を超える場合は、サーバーで時間方向をビン平均してから送る。ビニングした旨を表示する。
  - 成分・寄与・残差は0中心の発散カラースケールとする（既存`create_heatmap`の規則に従う）。
- トレンド：
  - ヒートマップのクリック、または波長・時間の2本のスライダーで地点を選ぶ。クリックとスライダーは同期し、ヒートマップ上に十字線を描く。
  - 「選択波長での強度 vs StepTime」と「選択時刻での強度 vs 波長」の2つのグラフを表示する。
  - トレンドの値はビニングしていない元の解像度からサーバーで切り出す。スライダーの操作はdebounce（150ms）してから取得する。
  - 複数ファイル、および「前処理済み vs 累積再構成」を重ねて表示できる。

### 4. スコア・ローディング

- スコア散布図：横軸PCm、縦軸PCn。1点が1ファイル。メタデータ列で色分けする。点をクリックすると、そのファイルを選択した状態でスペクトル探索を開く。
- ローディング散布図：成分mとnの係数を、特徴量ごとに波長単位で集計し、1点を1波長として打つ。集計方法は平均・RMS・絶対値平均から選ぶ（平均は符号の異なる寄与が打ち消し合うため）。
- 寄与率の表とスクリープロット（`get_explained_variance_table`）。
- 部分スコア軌跡：選んだファイル（複数可）について、PCm-PCn平面上の軌跡を描く。定義は「計算仕様」を参照。

### 5. T² / Q

- ファイル順の管理図（T²とQ、UCL線付き）。順序は`ui.default_order_by`の列、または画面で選んだ列とする。UCL超過点を強調する。
- T² × Q散布図（UCL線付き）。
- 点をクリックすると、そのファイルのQ寄与ヒートマップを開く。

## 計算仕様

以下は`feature_engineering/`（または`visualize/`のデータ準備関数）に独立した関数として追加し、単体テストする。

- 第k成分のみの寄与：前処理後の空間で$t_k \mathbf{w}_k$を求め、スケーリングの逆変換のうち中心化を除く部分（scaleの乗算）だけを適用して元のスケールに戻す。
- 部分スコア軌跡：1ファイルの前処理済み行$\mathbf{x}$に対し、fit時と同じ規則で補完・外れ値処理・スケーリングを適用して中心化した$\mathbf{z}$を得る。特徴量を`(Step, Sequence, StepTime)`の昇順に並べ、その順の累積和$\tau \mapsto \sum_{f \le \tau} z_f w_{m,f}$（成分nも同様）を求める。各`(Step, Sequence, StepTime)`点を1点とする。終点は通常のスコアと一致する（`whiten=False`の場合）。累積和は`float64`で計算する。
- Q寄与：1ファイルの前処理済み行に、fit時と同じ補完・外れ値処理・スケーリングを適用した$\mathbf{x}$（`pca.transform`へ渡す値）と、その再構成$\hat{\mathbf{x}}$との特徴量ごとの差の二乗$(x_f - \hat{x}_f)^2$とする（SPEC.md「Q統計量（SPE）仕様」と同じ空間）。再構成に使う成分数は、そのrunの`SpeConfig.cumulative_explained_variance`が選ぶ成分数に固定し、探索画面で選んだkは使わない。これにより、全特徴量（ビニング前）の寄与の総和は`scores.parquet`のQと一致する。
- ローディングの波長集計：`reshape_pca_components`の結果を波長でgroup_byし、平均・RMS・絶対値平均のいずれかを求める。
- ヒートマップのビニング：時間方向を等間隔のビンに分け、ビン内平均をとる。波長方向は間引かない（1200列程度を想定）。
- 実行前のメモリ見積もり：catalogの行数・波長数と前処理設定から特徴量数$F$を求め、$N \times F \times 8$Bの係数倍（初期値は3倍、実測で調整）を表示する。

## テスト方針

- `feature_engineering/`・`visualize/`の追加関数：単体テスト。
- `webui/services/`・`webui/jobs/`：小さなParquet・ダミーCSV・settings.tomlを`tests/`に用意して単体テストする。ジョブ関数はサブプロセスを介さずに直接呼ぶ。
- 実行器：実際に子プロセスを起動するテストで、次を確認する。
  - 実行中のジョブをキャンセルした後、待機中のジョブと新しく投入したジョブが成功する。
  - 待機中のジョブをキャンセルしても、実行中のジョブが成功する。
  - 子プロセスが異常終了したrunが`failed`になり、次のジョブが実行される。
- Q寄与：補完で欠損を埋めた行を含むデータで、ビニング前の寄与の総和が`scores.parquet`のQと一致する。
- `webui/routes/`：FastAPIの`TestClient`で、ステータスコードと返すpartialの主要要素を確認する。
- ブラウザ上の操作（クリック・スライダー）は自動テストの対象外とし、手動で確認する。

## マイルストーン

1. 基盤：settings、Workspace、DuckDB、catalogジョブ、データ選択画面
2. fitジョブ：前処理・PCA画面、run成果物、進捗・キャンセル、run一覧
3. スペクトル探索：元データ・前処理済み・成分のヒートマップ、トレンド、スライダー
4. 再構成系：単一成分・累積再構成・残差の表示
5. スコア・ローディング・部分スコア軌跡
6. T²/Q：管理図・散布図・Q寄与ヒートマップ

## 未決事項

- float32保存の影響の実データでの確認（float64保存とのT²・Q・再構成の比較スクリプトを用意し、手元で実行する）。
- メモリ見積もりの係数（実測で決める）。
- 部分スコア軌跡で`(Step, Sequence)`の並び順を、Step値の順でなく実際の時間順にする必要があるか。
