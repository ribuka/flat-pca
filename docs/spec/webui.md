# Web UI仕様（草案）

## スコープ

- ローカルPC上の1ユーザーが、Parquetファイル群の選択 → 前処理・Flatten-PCA → 可視化をブラウザから行うWeb UIを提供する。
- 構成はFastAPI + Jinja2 + htmxとし、グラフ描画はPlotly（ブラウザ側はPlotly.js）を使う。
- 将来、共有サーバー上で「同時実行1ユーザー」の運用へ移行できる構造にする（状態の集約・ジョブの単一実行）。認証・複数ユーザーの同時利用はv1の対象外とする。
- 対象外（将来検討）：
  - ブラウザからのファイルアップロード（Parquetのlazy読み込みにはランダムアクセス可能なストレージが必要なため、サーバーから見えるパスまたはURIを指定する方式とする）
  - `Time`で区間をつなげたヒートマップ（v1は`(Step, Sequence)`ごとの`StepTime`のみ）

## 構成

### 配置

```
src/flat_pca/
  feature_engineering/          データ生成（既存）。UIで必要な計算関数もここに追加する
  visualize/                    データ → go.Figure の純粋関数（Notebookからも使う）
  webui/
    __main__.py                 uv run -m flat_pca.webui [--settings <path>] [--reload] で起動
    reload_factory.py           --reload 用の引数なしのアプリfactory（設定ファイルのパスを環境変数から読む）
    app.py                      FastAPIアプリ生成、router登録、static mount
    settings.py                 settings.tomlの読み込み・検証
    settings_files.py           読み込む設定ファイルの選択（settings.local.toml優先）と読み込み
    settings_paths.py           パス設定値の展開（環境変数・{root}・相対パス）
    workspace.py                Workspace（DB接続・キャッシュ・ジョブ実行器）
    routes/                     画面・機能単位のAPIRouter（薄く保つ）
    services/                   routesから呼ぶ処理本体（catalog, runs, 表示用の行列・ビニング・キャッシュ）
    jobs/                       サブプロセスで実行するジョブ関数
    templates/{pages,partials}/ フルページとhtmxが差し替える断片
    templates/macros/           テンプレートから呼ぶJinjaマクロ（icon）
    static/                     vendor/htmx.min.js, vendor/material-symbols-outlined.woff2, app.js, app.css（plotly.min.jsはplotlyパッケージ同梱版を/static/vendor/で配信）
tests/webui/
```

- UI固有の依存（`fastapi`、`uvicorn`、`jinja2`、`duckdb`、`pydantic`）はoptional dependency `webui`とし、ライブラリ本体の依存に加えない。
- htmxは`static/vendor/`に同梱する。Plotly.jsはPythonの`plotly`パッケージに同梱された`plotly.min.js`を`/static/vendor/plotly.min.js`で配信し、Python側とバージョンを揃える。いずれもCDNに依存しない。
- アイコンはMaterial Symbols（Outlined）のうち使うアイコンだけを含むサブセットのwoff2を`static/vendor/`に同梱し、`templates/macros/icon.html`の`icon`マクロで表示する。アイコンを増やすときは`scripts/fetch_material_symbols.py`で取得し直す（手順は`static/vendor/README.md`）。
- テンプレートから読む static ファイルの URL には、ファイルの更新時刻を版として付ける（`?v={{ static_version(path) }}`。`templating.py::static_version`。plotlyパッケージから配信する`plotly.min.js`には、plotly の版`plotly_version`を付ける）。`app.js`などを変えたあと、ブラウザがキャッシュした古いファイルを新しいページに使わないようにするため。
- フォーム送信の解析に使う`python-multipart`も`webui`に含める。
- `--reload`のファイル監視に使う`watchfiles`も`webui`に含める。`uvicorn[standard]`にはしない（`httptools`などが入ると、`--reload`なしのHTTP実装やイベントループまで変わるため）。

### 開発用の自動リロード（`--reload`）

- `--reload`を付けると、`src/flat_pca`以下の`*.py`の変更で自動で再起動する。既定は無効で、開発用とする。
  - テンプレート・静的ファイル・settings.tomlは監視しない。テンプレートは次のリクエストで、静的ファイルはブラウザの再読み込みで反映される。settings.tomlは次の再起動で読み直す。
- uvicornのリロードはインポート文字列からワーカープロセスでアプリを作り直すため、`reload_factory.create_app_from_environment`をfactoryとして渡す。
  - `__main__.py`は`--settings`を検証したあと（不正なら終了コード2）、その絶対パスを自プロセスの環境変数`FLAT_PCA_WEBUI_SETTINGS`に入れてから`uvicorn.run`を呼ぶ。ワーカーはこれを受け継ぎ、factoryが`load_settings` → `create_app`する。
  - `--reload`なしの起動ではこの環境変数を読まない。
- リロードではlifespanの終了処理が走り、実行中のジョブは中断される（`Workspace.close()`で`cancelled`になる）。
- Windowsでは、uvicornはワーカーをCtrl+Cイベントで止めて再起動する。コンソールのないプロセスから起動すると、変更を検知してもワーカーが止まらず再起動しない。ターミナルから起動する。

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

起動時に`--settings`で指定する。省略時はカレントディレクトリの`config/settings.toml`を読む。`tomllib`で読み込み、pydanticで検証する。不正な場合は起動時にエラーとする。

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
explore_max_files = 20                # サイドバーで選べる表示ファイル数の上限
memory_poll_seconds = 5               # サイドバーのメモリ使用量を更新する間隔（秒）

[jobs]
artifact_dtype = "float32"            # "float32" | "float64"
memory_warn_gb = 16                   # 実行前見積もりがこれを超えたら確認を求める
```

- 型`category`・`number`・`datetime`のみを受け付ける。`format`は`datetime`のみに指定でき、省略時は形式を推定する。
- メタデータ列名に、結合キー列名と、ファイル一覧の列名（`stem`・`path`・`n_rows`・`n_steps`・`n_segments`）、run 成果物`samples.parquet`の列名`source`は使えない。
- settings.tomlと同じディレクトリに`settings.local.toml`（`<stem>.local<suffix>`）があれば、settings.tomlの代わりにそちらだけを読む（マージはしない）。`*.local.toml`はgit管理外とし、個人環境のパスを置く。
- `workspace.dir`・`data.root`・`metadata.csv`では次の順に展開する。
  1. 環境変数`%NAME%`・`${NAME}`を置換する。未設定の変数はエラーとする。
  2. `{root}`を展開済みの`data.root`に置換する（`workspace.dir`・`metadata.csv`のみ。例：`csv = "{root}/meta.csv"`）。それ以外の`{name}`はエラーとする。
  3. 先頭の`~`をホームディレクトリに展開する。
- `workspace.dir`・`data.root`・`metadata.csv`の相対パスは、settings.tomlのあるディレクトリ基準で解決する。未知のキーはエラーとする。
- メタデータの結合キーは`Path.stem`とする。`data.root`配下でstemが重複する場合はcatalog構築をエラーとする（UI経由の実行では`stem_uniqueness="error"`を使う）。
- メタデータCSVの結合キーが空または重複する場合、取り込む列が無い場合、値を型に変換できない場合はcatalog構築をエラーとする。
- CSVに無いファイル、ファイルが存在しないCSV行は件数と一覧をcatalog画面に警告表示し、処理は継続する。

## Workspace と DB

- 状態は1つの`Workspace`オブジェクト（DuckDB接続、表示用キャッシュ、ジョブ実行器）に集約し、グローバル変数に分散させない。
- 表示用キャッシュ（`services/display_cache.py`）は、読み込んだ元データ（`StepTime`列を付けたファイル全体。パス・サイズ・更新時刻をキーとする）、transform runの成果物（`X.npy`はmemmapのまま。モデル側とデータ側のrunディレクトリの組`RunDirs`をキーとする）、モデル画面が使う fit run の特徴量と`components.npy`（memmapのまま。fit run のディレクトリをキーとする）をそれぞれLRUで保持する。
- DuckDBファイルは`{workspace.dir}/flatpca.duckdb`とする。書き込みは親プロセスのみが行う。
- スキーマに版管理・マイグレーションは持たない。スキーマを変えたときはWorkspaceを作り直す。`file_metadata`だけは、settings.tomlの列定義（列名・順序・SQL型）と食い違う場合に起動時に空で作り直す（次のcatalog更新で埋まる）。
- テーブル：
  - `files`：`stem`、`path`、`size`、`mtime_ns`、波長数・最小・最大、`Time`の最小・最大、行数
  - `segments`：`stem`、`Step`、`Sequence`、行数、StepTime最大値
  - `file_metadata`：`stem` + settings.tomlで定義した列
  - `runs`：`run_id`、ジョブ種別（`catalog`・`fit`・`transform`）、作成日時、状態（`queued`/`running`/`succeeded`/`failed`/`cancelled`）、設定JSON、対象ファイル数、特徴量数、成分数、所要時間、エラーメッセージ、成果物ディレクトリ
- catalogは`path`・`mtime`・`size`のいずれかが変化したファイルのみを再走査する差分更新とする。走査は`Time`・`Step`・`Sequence`列とスキーマのみを読む。

## ジョブ実行

- ピークメモリが大きくなり得る処理はサブプロセスで実行する。対象は次のとおり。
  - catalogの構築・更新
  - 前処理・flatten・PCA fit（1つのfitジョブ。fit 対象のスコアは付けない）
  - 学習済みモデルによる前処理・flatten・transform・スコア付与（1つのtransformジョブ）
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
- ジョブ本体は通常の関数（例：`jobs/fit_run.py::run_fit(config, run_dir)`・`jobs/transform_run.py::run_transform(config, run_dir)`）とし、サブプロセスを介さず単体テストできるようにする。

## run 成果物

1回の「前処理 → PCA fit」の結果一式を`{workspace.dir}/runs/{run_id}/`に保存する。

| ファイル | 内容 | 形式 |
| --- | --- | --- |
| `config.json` | 前処理・PCA・T²/Q設定、対象ファイル一覧（ファイルごとのメタデータを含む）、`artifact_dtype` | JSON |
| `features.parquet` | 特徴量名`feature`と座標`(wavelength, Step, Sequence, StepTime)`、行順は`X.npy`の列順 | Parquet |
| `samples.parquet` | `source`、`stem` + ファイルメタデータ（settings.tomlの型）、行順は`X.npy`の行順 | Parquet |
| `X.npy` | 前処理・flatten・sparse列除去後、補完前の行列（欠損はNaN） | `.npy` |
| `components.npy` | `pca.components_`（`n_comp × n_features`） | `.npy` |
| `pca_state.npz` | PCA（mean・explained_variance等）と前処理3段階の状態（下記） | `.npz` |
| `progress.json`・`log.txt` | 進捗（`preprocess`・`fit`・`save`の3段階）・ログ | JSON・テキスト |

- fit run はスコア・T²・Qを持たない（`scores.parquet`を出さない）。fit 対象を含め、スコア・T²・Qは transform run だけが持つ（transform 画面で「Run transform」を押したときにだけ作る）。
- 巨大な数値行列は`.npy`とし、親プロセスは`np.load(..., mmap_mode="r")`で開いて必要な行だけを読む。表形式のデータはParquetとする。
- `X.npy`・`components.npy`は`jobs.artifact_dtype`で保存する（既定`float32`）。読み込み後の計算（スケーリング、スコア、再構成、残差、Q、累積和）はすべて`float64`へ変換してから行う。
- `.npy`/`.npz`から`PcaModel`を復元する関数を`pca/serialization.py`に追加する（`build_pca_state`・`parse_pca_state`、`PcaModel.to_pca_state`・`PcaModel.from_pca_state`）。既存のJSONシリアライズは変更しない。
- `pca_state.npz`は前処理の段階ごとに配列を持つ。特徴量ごとの値は`features.parquet`の行順に並べ、値を持たない段階は空の配列とする。
  - PCA：`n_component`、`pca_column_names`、`pca_mean`、`pca_explained_variance`、`pca_explained_variance_ratio`、`pca_singular_values`、`pca_n_samples`、`pca_noise_variance`、`pca_whiten`
  - 補完：`impute_strategy`、`impute_values`、`impute_kmeans_n_clusters`（kmeans以外は0）、`impute_kmeans_centroids`（`n_clusters × 特徴量`）
  - 外れ値：`outlier_strategy`（無効は空文字列）、`outlier_iqr_multiplier`、`outlier_outlier_lower`・`outlier_outlier_upper`・`outlier_winsor_lower`・`outlier_winsor_upper`
  - スケーリング：`scaling_strategy`、`scaling_centers`、`scaling_scales`（`"none"`の場合は空）
- 成果物に形式の版は持たせない。読み込めない成果物（形式が古い・壊れている）はエラーとして表示し、再実行を促す。

### transform run の成果物

学習済みの fit run のモデルで transform 対象（fit 対象と同じファイル、または catalog から選んだファイル）を transform した結果を`{workspace.dir}/runs/{run_id}/`に保存する。モデル（`features.parquet`・`components.npy`・`pca_state.npz`）は複製せず、元の fit run のものを使う。

| ファイル | 内容 | 形式 |
| --- | --- | --- |
| `config.json` | 元の fit run の`fit_run_id`・`fit_run_dir`、対象ファイル一覧（ファイルごとのメタデータと、投入時のサイズ`size`・更新時刻`mtime_ns`を含む）、fit run の前処理・T²/Q設定、`artifact_dtype` | JSON |
| `samples.parquet` | fit run と同じ形式 | Parquet |
| `X.npy` | 前処理・flatten 後、補完前の行列。列は fit run の`features.parquet`の順（欠損はNaN） | `.npy` |
| `scores.parquet` | `source`、`pca-1..k`、T²（`mahalanobis_*`）・Q（`spe_*`）列。`impute_strategy="drop"`で除いた行は含まない。T²・QのUCLは fit run のモデル（`MahalanobisConfig`・`SpeConfig`）から求めるため、同じ fit run の transform run では同じ値になる | Parquet |
| `progress.json`・`log.txt` | 進捗（`preprocess`・`transform`・`save`の3段階）・ログ | JSON・テキスト |

- 前処理は fit run の設定（`config.json`の`preprocess`）で行う。ただし、列は fit run の特徴量で決まるため、transform 対象の格子での間引き（`t_downsampling_stride`・`w_downsampling_stride`）と、transform 対象の欠損率による列の除去（`max_null_ratio`）は行わない。間引きは格子点を選ぶだけで、平滑化・正規化の後に行うため、間引かずに fit run の特徴量を選べば、格子が同じときは fit と同じ値になる。要素ごとの強度変換（`intensity_transform`）は、fit run の特徴量を選んだ後に行う（`jobs/transform_run.py::transform_target_intensity`）。fit で間引いて使わなかった値が`log1p`の定義域外でも、transform は失敗しない。
- flatten 後の列を fit run の特徴量にそろえる（`jobs/transform_run.py::align_features`）。transform 対象の`(Step, Sequence, StepTime)`の格子や波長が fit と違い、fit の特徴量に足りない列は欠損値とし、fit run の補完で埋める。fit run にない列は捨てる。fit run の特徴量が1つもない場合はエラーとする。
- `impute_strategy="drop"`の fit run では、欠損を含む transform 対象の行は`X.npy`・`samples.parquet`に残し、`scores.parquet`には含めない（fit の補完と同じ扱い）。すべての行が除かれる場合はエラーとする。
- 表示側は`services/run_dirs.py::run_dirs`で、モデル側（fit run）とデータ側（transform run）のディレクトリを解決し、`services/fit_artifacts.py::load_display_artifacts`で読む。モデル画面は fit run の特徴量と`components.npy`だけを`load_model_artifacts`で読む。transform run の検証は`services/transform_artifacts.py::register_transform_result`で行う。

## 画面

### 共通レイアウト

- 左にサイドバーを置き、各画面へのナビゲーション（未実装の画面は無効表示）、表示ファイルの選択、catalogの状態・fit 対象のファイル数を表示する。表示する run はサイドバーでは選ばない（transform 画面で選ぶ）。
- サイドバーは上から、折りたたみボタンとメインページの幅の切り替え、ナビゲーション、表示ファイルの選択、catalogの状態・fit 対象のファイル数、メモリ使用量と version の順に並べる。中身（ナビゲーションから catalog の状態まで）だけをスクロールさせ、上端と下端は常に表示する。
  - 折りたたみボタン（Material Symbols の`left_panel_close`・`left_panel_open`）は、折りたたむとボタンだけを残してサイドバーを狭める。ボタンの画面上の位置は折りたたみの前後で変えない。折りたたんでも、メインページはそのとき選ばれている run とファイルで表示したままとする。
  - メインページの幅は segmented control で「Compact」（最大幅を固定して中央に寄せる）と「Wide」（ウィンドウの幅に合わせる。既定）から選ぶ。
  - 折りたたみ状態と幅は`localStorage`に保存し、ページを移っても保つ。最初の描画の前に`<html>`の`data-sidebar`・`data-width`へ反映し、ちらつかせない。「戻る」で bfcache から復帰したページ（`pageshow`の`persisted`）にも、離れた後に変えた保存値を反映する。幅や折りたたみを変えたら、Plotly の図を`Plotly.Plots.resize`で新しい幅に合わせて描き直す。
  - 下端には、Web UI のサーバープロセスのメモリ使用量（RSS）、マシン全体のメモリ使用量 / 総量（と使用率）、flat-pca の version（`VERSION`ファイルの値。インストール済みパッケージのメタデータから読む）を表示する。メモリ（`/sidebar/system`。`psutil`で取得）は`ui.memory_poll_seconds`秒（既定 5）ごとにポーリングする。
- サイドバーの状態は、catalog更新の開始・完了とファイル集合の選択（`catalog-started`・`catalog-updated`・`selection-updated`イベント）で再取得し、catalog実行中は2秒ごとにポーリングする。
- 用語：データ選択で選ぶファイル集合を「fit 対象」（画面では Fit target）、サイドバーで選ぶファイルを「表示ファイル」（画面では Shown files）と呼び、画面の文言で区別する。
- 画面の文言（ナビゲーション・見出し・ラベル・ボタン・選択肢・プレースホルダー・表の列名・状態・警告・エラーメッセージ・図のタイトル）は英語とする。詳細説明（`.muted`の注記）だけを日本語で書く。
- 各画面の内容は、データ選択と同じく折りたためるカード（`<details class="card">`、既定は開いた状態）に分ける。先頭のカードは画面名を見出しにする。折りたたんだカードを開いたら、Plotly の図を`Plotly.Plots.resize`でカードの幅に合わせて描き直す。
- 表示条件の入力欄は、それが変える図のカードに置く。ページを読み直すフォームは1つだけとし、ほかのカードの入力欄はHTMLの`form`属性でそのフォームに属させる（自動送信・リセットも同じフォームで行う）。
- 表示する transform run と表示ファイル：
  - T² / Q・スペクトル探索・スコアは、transform 画面で選んだ transform run（表示する run）と、サイドバーで選んだ表示ファイルを使う。モデル画面は transform 画面で選んだモデル（fit run）を使う。各画面のメインには run とファイルの選択欄を置かない。
  - 表示する run は、成功した transform run から選ぶ。何も選んでいないとき、または選んでいた run が成功していない（実行中・失敗・消えた）ときは、最新の成功した transform run を使う。成功した transform run がなければ、これらの画面は transform を先に実行する旨（No succeeded transform run. Run a transform on Transform first.）を表示する。
  - 表示する run のモデル（寄与率・ローディング・PCA成分・前処理の状態）は元の fit run のものを、ファイル（元データ・前処理済み・再構成・スコア・T² / Q）は transform run のものを使う。T² / Q の UCL は fit run のもの（`MahalanobisConfig`・`SpeConfig`）とする。
  - サイドバーの表示ファイル（`/sidebar/selection`。どの画面でも常に表示する）の選択肢は、表示する run の transform 対象（`samples.parquet`の`stem`）を`natural_keys`の順に並べたもの。run がなければ空とする。検索ボックスで絞り込めるチェックボックスのリストとし、複数選べる。上限は`ui.explore_max_files`件（既定 20）で、上限に達したらほかのチェックボックスを無効にしてその旨を表示する。サーバーも上限を超えた分（選択肢の順で後ろのもの）と選択肢にないファイルを捨てる。fit・transform が終わったら（`fit-updated`・`transform-updated`イベント）取得し直す。
  - 表示する run を変えると、表示ファイルの選択をすべて解除する。
  - 表示ファイルの変更は、サイドバーを描いたときの run も送る。その後に run が変わっていれば（別のタブなど）、選択を変えずに run の変更（`{"changed": "run"}`）を知らせ、サイドバーとメイン部分を今の run で取得し直させる。使う run の解決・一致の確認・選択の更新は、`ViewSelection.transaction()`のロックの中でまとめて行い、ほかの要求による変更が途中に入らないようにする。
  - 選択はデータ選択の選択（`services/selection.py::FileSelection`）と同じく、サーバー側の workspace（表示する run と表示ファイルは`services/view_selection.py::ViewSelection`、transform 画面のモデルとチェックボックスは`services/transform_settings.py::TransformSettings`）に持つ。画面を移っても、ブラウザで再読み込みしても残り、ブラウザのタブ間で共有される。
  - 選択を変えると、サーバーは`HX-Trigger`で`view-selection-changed`イベント（表示する run は`{"changed": "run"}`、表示ファイルは`{"changed": "files"}`、transform 画面のモデルとチェックボックスは`{"changed": "model"}`）を返し、ページは読み直さない。ブラウザ（`app.js`の`refreshView`）は、サイドバーの選択（`#view-selection`）と、今の画面がその選択を使うときはメイン部分（`<main class="content">`の中身）を今の URL（クエリを含む）で取得し直し、両方が届いてからまとめて差し替える。URL と履歴は変えない。
    - メイン部分を差し替える選択は、`base.html`の`<main>`の`data-view-swap`に画面ごとに書く（`view_swap`ブロック）。スペクトル探索・スコアは run と表示ファイル（`run files`）、T² / Q は run だけ（`run`。表示ファイルを使わないため）、transform・モデルはモデルだけ（`model`。transform 画面では、表示する run を変えても送信メッセージを失わないよう、transform run 一覧だけを`view-selection-changed`で取得し直す）、データ選択と前処理・PCA は差し替えない（編集中のフォームを失わないため）。
    - これらの画面のルートは`routes/view_page.py::render_view_page`で描き、htmx の要求（`HX-Request: true`）にはメイン部分だけを返す（`base.html`が`main_only`で切り替える）。応答には`Vary: HX-Request`を付ける。
    - 差し替える前に、メイン部分の Plotly の図を`Plotly.purge`で破棄し、差し替え前に始まったトレンドの取得が差し替え後の図に描かないようにする。差し替えたあと（`htmx:afterSettle`）とページを開いたとき（`pageshow`）に、同じ関数（`initMain`）でメイン部分を初期化する。
    - 表示ファイルを変えたときは、ファイルリストのスクロール位置を差し替えの前後で保つ。run を変えたときはリストの中身が変わるので先頭に戻す。検索ボックスの文字列はタブごとに`sessionStorage`へ保存し、差し替えやページの読み直しのあとも保つ。
    - 待つ間は下記のオーバーレイを表示する。キャンセルでは htmx の要求（`htmx:abort`）と取得を止め、サイドバーをサーバー側の選択で取得し直す（メイン部分は差し替えない）。取得に失敗したときは何も差し替えずにオーバーレイを閉じる。
  - スコア散布図の点のクリックは、`/sidebar/selection/files/add`に図の run とファイルを送り、そのファイルを表示ファイルに追加する。上限に達しているとき、または図を描いた後に表示する run が変わったとき（別のタブなど）は追加せず、警告を表示する。通信に失敗したときも警告を表示する。追加したら、上記と同じくサイドバーとメイン部分を差し替える。
- フォーム内でhtmxの要求を出す要素は、フォームの`hx-target`を継承しないよう`hx-target`を明示する。
- 表示条件を変えるとページ全体を読み込み直すフォーム（`data-auto-submit`。モデル・スペクトル探索・スコア・T² / Q）は、送信から次のページの表示までオーバーレイを表示する。
  - 送信と同時に操作をできなくし（`inert`）、300 ms 後に画面をグレーアウトして中央にスピナー・経過時間（秒）・キャンセルボタンを表示する。短い処理ではちらつかない。
  - キャンセルは読み込みを中止し（`window.stop()`）、フォームを表示中の条件（フォームの初期値）に戻す。サーバー側の計算は止まらず、最後まで続く。
  - 「戻る」で表示したページ（bfcacheからの復元を含む）では、オーバーレイを消し、フォームを表示中の条件に戻す（`pageshow`）。
  - オーバーレイは`app.js`の`busyOverlay.start(onCancel, options)`・`busyOverlay.update(elapsedS, source)`・`busyOverlay.stop()`で、ほかの待ち時間にも使える。`options`では表示する文言・経過時間の起点・持ち主（`owner`）と、キャンセル後も持ち主が閉じるまで表示を続けるか（`keepOnCancel`）を指定する。

### 1. データ選択

- catalogのファイル一覧を表示する（ファイル名、メタデータ列、Step/Sequence数）。
  - 列名のクリックでその列の昇順にソートし、もう一度クリックすると降順にする（以降トグル）。ソート中の列と向きを列名の横に▲ / ▼で示す。
  - 列名の下に列ごとのフィルタを置く。ファイル名は部分一致、category列はプルダウン、数値・日時の列は下限と上限。Step数・(Step, Sequence)数・行数にはフィルタを置かない。
  - ソート・フィルタ・ページングはサーバー側（`/catalog/files`の`sort`・`order`・`page`・各フィルタのクエリ）で行う。ソートやフィルタを変えると1ページ目に戻す。
  - 表はヘッダと12行ほどの高さを上限とし、はみ出す行は表の中でスクロールする（ヘッダは固定）。1000行ごとにページを分け、表の下にページの移動と「a–b of N」を表示する。
  - 選択はページやフィルタをまたいで保持する（表の外の1つのhiddenのinputにJSON配列で持つ）。ヘッダのチェックボックスは、表示していないページも含めて今のフィルタに合う全行を対象に選択・解除し、その状態（チェック・一部・なし）を表示する。
  - 表の下の`Select`で、全ページの選択をまとめて保存する。選択はJSON配列の1つのフォームフィールドとして送り、フォームのフィールド数の上限（Starletteの既定で1000）に掛からないようにする。
- 「Update catalog」でcatalogジョブを起動する。
- 選択したファイル集合を次の画面へ渡す。

### 2. 前処理・PCA

- `PreprocessConfig`の各フィールドに対応するフォームを持つ。
  - `target_steps`はcatalogのStep一覧から複数選択する（既定はすべて）。
  - `wavelength_range`・`*_normalization_range`はチェックボックスで有効にする（既定は無効）。入力欄の初期値と範囲はcatalogの波長・`Time`の最小・最大から決める。
  - `intensity_transform`（`"none"`・`"sqrt"`・`"log1p"`・`"asinh"`）と`intensity_transform_scale`を設定できる。`"log1p"`の定義域エラーは実データを読むまで分からないため、fitジョブ内の`ValueError`としてrunを`failed`にし、そのメッセージを表示する。
- PCA設定：`n_component`の既定（空欄）は「累積寄与率0.99に達する成分数、上限1000」とし、UIで変更できる。上限1000まで成分をfitしてから先頭の成分だけを残し（`truncate_pca_model`）、捨てた成分の分散は`noise_variance_`へ平均として畳み込む。`impute_strategy`（`"kmeans"`のときは`impute_kmeans_n_clusters`）、`scaling_strategy`（`"none"`・`"z-score"`・`"minmax"`・`"robust"`・`"pareto"`）、`MahalanobisConfig`・`SpeConfig`の`cumulative_explained_variance`と`alpha`を設定できる。外れ値処理は`flatten_pca`と同じく使わない（`outlier_strategy=None`）ため、画面に出さない。
  - 成分数を打ち切るとQのUCL（$\theta_2$・$\theta_3$）の精度が下がる（SPEC.md「Q統計量（SPE）仕様」）。画面に注記する。
  - `scaling_strategy`が`"none"`以外ではNumPyの高速経路を使わず、fitが遅くなり必要メモリも増える。画面に注記する。
- 実行前に、catalogから特徴量数と必要メモリを見積もって表示する（フォームの変更ごとに更新する）。`jobs.memory_warn_gb`を超える場合は確認欄へのチェックを求め、チェックが無い送信はエラーとする。
- 送信は`hx-post`とし、`PreprocessConfig`等の`ValueError`は該当フィールドの横にpartialで表示する。UI経由の実行では`stem_uniqueness="error"`を使う。
- カードは上から、画面名（fit 対象のファイル数）、設定グループ（Target Steps・Preprocessing・PCA・T² / Q）、Memory estimate（見積もり）、Run status（実行状況）、Runs（run一覧）とする。実行ボタン（Run fit）は Run status のカードの先頭に左揃えで置き、`form="fit-form"`でフォームを送信する。ブラウザの入力検証で不正な欄があれば、その欄を含む折りたたんだグループを開く。
- fitのrunが待機中・実行中の間は、上記のオーバーレイで画面全体を覆って操作できなくする。実行中のrunがある状態でページを開いた（読み直した）ときも同じ。
  - オーバーレイには、スピナー・経過時間（runの作成時刻からの秒数。読み直しても作成時刻から数える）・段階と進捗（`progress.json`）・残り時間の見込み・キャンセルボタンを表示する。
  - 残り時間は`経過 × (total − done) / done`とする（`Progress.remaining_s`）。fitの進捗は4段階を通した完了数のため、全段階の残り時間である。`done = 0`の間は「estimating」と表示する。
  - 経過時間と残り時間はサーバーが計算して状態のpartialに埋め込み（1秒ごとのポーリングで更新）、ブラウザはポーリングの間も経過時間を数え進める。
  - キャンセルは`POST /runs/{run_id}/cancel`を呼び、runが終わるまで「Cancelling…」と表示し続ける。要求が失敗したとき（通信の失敗や409以外のエラー）は、その旨を表示してキャンセルボタンを再び押せるようにする。
  - runが終わったら（成功・失敗・キャンセル）オーバーレイを閉じ、実行状況のカードを開いて状態を表示する。成功時は実行ボタンの横に`check_circle`を表示する。
- 実行中は実行状況にも進捗とキャンセルボタンを表示する。過去のrun一覧から再訪でき（`/fit?run={run_id}`）、そのrunの設定をフォームに読み込んで状態を表示する。

### 3. transform

- `/transform`。transform に使うモデル（成功した fit run。新しい順。表示は`run_id（作成日時）`）の選択欄と、`use same data for fit`のチェックボックスを置く。モデルの既定は最新の fit run（選んでいた fit run がなくなったときも最新に戻す）、チェックボックスの既定はチェックあり。どちらも workspace（`TransformSettings`）に持ち、変えると`POST /transform/settings`で保存して`{"changed": "model"}`を知らせ、メイン部分を差し替える。チェックなしのときの表の選択も、このとき workspace に保存する。
- transform 対象を選ぶ表と「Run transform」ボタンは、チェックの有無にかかわらず常に表示する。
  - 表はデータ選択と同じもの（`/catalog/files`の絞り込み・ソート・ページング、ページやフィルタをまたいだ選択、ヘッダのチェックボックスでの一括選択）とする。
  - チェックありのときは、モデルの fit 対象（fit run の`config.json`のファイル）を選んだ状態で表示し、表のチェックボックスを無効にしてグレーアウトする（`#file-table`の`data-locked`。絞り込み・ソート・ページングは使える）。
  - チェックなしのときは、catalog から自由に選べる。選択は送信時に workspace（`Workspace.transform_selection`）へ保存し、画面を開き直しても残す。
- fit を実行しただけでは transform しない。「Run transform」（`POST /transform`）を押したときにだけ、選んだモデルで transformジョブ（種別`transform`）を投入する。モデルとチェックの有無は、画面を描いたときのもの（設定フォームの値）を要求に含めて使い、別のタブで設定を変えても、その画面で選んだものとは別のモデル・対象を transform しない。要求のモデルがもう成功した fit run でなければ、画面の読み直しを促すエラーを表示する。チェックありのときは fit 対象を、チェックなしのときは表で選んだ catalog のファイルを対象とする。fit run がない・対象がないときは、ボタンの横にエラーを表示する。
  - チェックの有無を問わず、同じ fit run で同じ対象ファイルの集合（順序は問わない）を transform した成功済みの transform run があり、その後どのファイルも変わっていなければ（`config.json`に保存したファイルごとのパス・サイズ・更新時刻`size`・`mtime_ns`が今と同じなら）、ジョブを作らずにその run を表示する run にし、その旨をボタンの横に表示する（`services/transform_targets.py::find_transform_run`）。チェックなしで fit 対象と同じファイルを手で選んだ場合も同じ。
  - 投入した transform run は、その時点で表示する run に選んでおき、成功したら表示する（実行中・失敗したときは最新の成功した transform run を表示する）。
- 実行中は前処理・PCA画面と同じオーバーレイ（「Running the transform…」、経過時間、段階と進捗、残り時間、キャンセル）と実行状況を表示し、`/transform/runs/{run_id}/status`をポーリングする。終わったら`transform-updated`で transform run 一覧とサイドバーの表示ファイルを取得し直す。
- transform run 一覧（`/transform/runs`）は、表示中の run（元の fit run とファイル数）と、run・元の fit run・作成日時・状態・ファイル数・所要時間を新しい順に表示する。成功した run の「Show」（`POST /transform/show`）で、その run をスペクトル探索・スコア・T² / Q とサイドバーの表示ファイルで使う run にする。

### 4. モデル

- fit runを選んだ時点で決まり、ファイルの選択で変わらない図をまとめる。fit run は transform 画面で選んだモデルを使い、表示ファイルの選択は使わない。fit run がなければ、その旨を表示する。
- 選択項目：ローディングの成分番号m・n（既定はPC1・PC2。範囲外は既定に戻す）、ローディングの集計方法、ヒートマップの値（PCA成分kか前処理パラメータ。既定はPCA成分k）、成分番号k（既定は1。範囲外は既定に戻す）、`(Step, Sequence)`（`features.parquet`から求める。既定は先頭）。
  - 選択は`/model?x={m}&y={n}&aggregation=…&view=…&k=…&segment={Step}:{Sequence}`のクエリで表す。
  - ローディングの成分番号m・n・集計方法は Loadings のカードに、ヒートマップの値・成分番号k・`(Step, Sequence)`は Heatmap のカードに置く。
- 表示する図（上から順に）：
  1. 寄与率の表とスクリープロット（`get_explained_variance_table`）。
  2. ローディング散布図：成分mとnの係数を、特徴量ごとに波長単位で集計し、1点を1波長として打つ。集計方法は平均・RMS・絶対値平均から選ぶ（既定はRMS。平均は符号の異なる寄与が打ち消し合うため）。
  3. ヒートマップとトレンド：「ヒートマップの値」で選んだ、特徴量（時刻 × 波長）ごとの値を`(Step, Sequence)`ごとのStepTime × 波長に並べる（`feature_segment_matrix`）。ヒートマップ・トレンドの描き方（スライダーとクリック、十字線、ビニング）はスペクトル探索と同じとする。トレンドは`/model/trend?run=…&view=…&k=…&segment=…`でビニング前の行列から切り出す。ページ表示時の行列を、run・値（view）・k・`(Step, Sequence)`をキーに表示キャッシュへ保持し（スペクトル探索と同じキャッシュ）、外れている場合は解決し直す。
     - PCA成分k（`view=component`）：`components.npy`の1行。0中心の発散カラースケールで描く。
     - 前処理パラメータ：`PcaModel`が保存している特徴量ごとの値（`model.columns`は`features.parquet`の行順）。連続カラースケールで描く（カラーバーは`value`）。
       - 中心化の平均（`mean`）：`pca.mean_`。スケーリング後の値の平均。常にある。
       - スケーリングのcenter・scale（`scaling_center`・`scaling_scale`）：`scaling_strategy`が`"none"`以外のとき。
       - 補完値：`impute_strategy="median"`の中央値（`impute_median`）、`"kmeans"`のクラスタ重心（`kmeans_centroid_{c}`。cは1始まり）。
       - 外れ値処理の閾値（`outlier_lower`・`outlier_upper`）とwinsorizeの上下限（`winsor_lower`・`winsor_upper`）：外れ値処理をしたとき。Web UIのfitは`outlier_strategy=None`のため、通常は持たない。
     - 選択肢にはそのrunが持つ値だけを出す。runが持たないパラメータを`view`で指定した場合は、持たない旨を表示してPCA成分kを表示する。前処理パラメータとして不正な`view`は400とする。

### 5. スペクトル探索

- 表示対象：
  - 元データ（Parquetから読む）
  - 前処理済み（`X.npy`の1行をreshape）
  - 第k成分のみの寄与（元のスケール）
  - 先頭1..k成分の累積再構成（`PcaModel.reconstruct`と同じ計算）
  - 残差（補完・外れ値処理後の前処理済み − 累積再構成、元のスケール）
  - Q寄与（特徴量ごとの残差の二乗。定義は「計算仕様」を参照）
- 「前処理済み」は`X.npy`の値をそのまま表示し、欠損はNaN（空白セル）として示す。再構成・残差・Q寄与は、fit時と同じ補完・外れ値処理を適用した値から計算するため欠損を含まない。`impute_strategy="drop"`のrunで欠損を含むファイルは除外されて再構成できないため、その旨を表示する。強度変換の逆変換は行わない（「元のスケール」は`X.npy`と同じ強度変換後の空間を指す）。
- run は transform 画面で選んだ表示する run、ファイルはサイドバーで選んだものを使う。どのビューも（元データを含めて）、その run の transform 対象（`samples.parquet`のファイル。元データは`source`のパスから読む）を表示する。run がなければ、元データも表示せず、transform を先に実行する旨を表示する。
- 選択項目：表示対象（ビュー）、ヒートマップのファイル（Heatmap file）、`(Step, Sequence)`、成分番号k。
  - ビュー・`(Step, Sequence)`・成分番号kは画面名のカードに、ヒートマップのファイルは Heatmap のカード（ヒートマップとトレンド）に置く。
  - サイドバーで選んだファイルを`natural_keys`の順に並べ、すべてのファイルをトレンドに重ねる。ファイルを選んでいなければ先頭のファイルを表示する。
  - ヒートマップには「Heatmap file」で選んだ1つを描く。選択肢は表示できたファイル（`(Step, Sequence)`を持たないファイルや、補完方法 drop で除外されたファイルは含めない）で、既定は先頭のファイル。選んだファイルが選択肢から外れたら先頭に戻す。トレンドではヒートマップのファイルを先頭に、残りを選択肢の順に重ねる。ヒートマップにはタイトルを付けない。
  - 一度に表示するファイルは`ui.explore_max_files`件（既定 20）までとする（サイドバーの上限と同じ）。トレンドの要求などで超えて指定された場合は選択肢の順で先頭から上限までを表示し、ページでは表示しなかったファイルの件数と一覧を表示する。準備済みの行（再構成・残差・Q寄与で使う、補完・外れ値処理後の行）の表示キャッシュも同じ件数を保持するため、上限までのファイルなら表示し直しても`prepare_rows`は再実行されない。
  - `(Step, Sequence)`の選択肢は、元データではヒートマップのファイルの元データ、それ以外では`features.parquet`から求める。選んだ`(Step, Sequence)`を持たないファイルはトレンドから除き、その旨を表示する。
  - 選択は`/explore?view=…&heatmap_file=…&segment={Step}:{Sequence}&k=…`のクエリで表す。トレンドの要求（`/explore/trend`）は、描いたページの run とファイルも`run=…&file=…`で指定する。
  - Q寄与（`view=q_contribution`）は成分数をrunのQの設定で固定するため、kを選ばない（再構成に使う成分数を画面に表示する）。
- 前処理済みでは、そのrunの`intensity_transform`（と`intensity_transform_scale`）を画面に表示する。
- ヒートマップ：横軸波長、縦軸`StepTime`（0を下）。
  - セル数が`ui.heatmap_max_cells`を超える場合は、サーバーで時間方向をビン平均してから送る。ビニングした旨を表示する。
  - 寄与・残差は0中心の発散カラースケールとする（既存`create_heatmap`の規則に従う）。
- トレンド：
  - ヒートマップのクリック、または波長・時間の2本のスライダーで地点を選ぶ。クリックとスライダーは同期し、ヒートマップ上に十字線を描く。
  - 「選択波長での強度 vs StepTime」と「選択時刻での強度 vs 波長」の2つのグラフを表示する。
  - トレンドの値はビニングしていない元の解像度からサーバーで切り出す（`/explore/trend`）。各ファイルは、要求された波長・`StepTime`に最も近い自身の格子点で切り出す。ページ表示時に解決したビューのビニング前の行列を、表示条件（ビュー・fit run・ファイル・ヒートマップのファイル・(Step, Sequence)・k）をキーに表示キャッシュへ保持し、トレンドはそこから切り出す（最大 8 ビュー・合計 256 MiB）。キャッシュから外れている場合や、ページを経ずに要求された場合は、ビューを解決し直す。ページを表示し直すと、保持した行列は解決し直した値で置き換わる。スライダーの操作はdebounce（150ms）してから取得する。
  - 複数ファイル、および「前処理済み vs 累積再構成」を重ねて表示できる（累積再構成の表示では、各ファイルの前処理済みを常に重ねる）。

### 6. スコア

- run は transform 画面で選んだ表示する run、軌跡のファイルはサイドバーで選んだものを使う。
- 選択項目：成分番号m・n（既定はPC1・PC2。範囲外は既定に戻す）、色分けの列。
  - 選択は`/scores?x={m}&y={n}&color=…`のクエリで表す。
  - 選択欄と表示ボタンは Score scatter のカードに置く。カードは上から、画面名・Score scatter・Selected points（選んだ点の表）・Partial score trajectories とする。
- スコア散布図：横軸PCm、縦軸PCn。1点が1ファイル（`scores.parquet`の値。`impute_strategy="drop"`で除いたファイルは含まない）。メタデータ列で色分けする。
  - 色分けの既定は`ui.default_color_by`とし、「none」も選べる。数値の列は連続カラースケール、それ以外の列は値ごとの系列（欠損は1つの系列）とする。
  - 点をクリックすると、そのファイルをサイドバーの表示ファイルに追加し、今の画面にとどまる（ページは読み直さず、サイドバーとメイン部分を差し替える）。上限に達していれば追加せず、警告を表示する。
  - 点のクリックと範囲選択（box / lasso）で、選んだ点を「選んだ点の表」に表示する。表の列は、ファイル名・メタデータ列・表示中のPCmとPCnのスコアとする。クリックした点の行は、ファイルを追加した後に差し替えた表へそのまま渡す。
- 部分スコア軌跡：サイドバーで選んだファイル（`natural_keys`の順。未選択なら先頭のファイル）について、PCm-PCn平面上の軌跡を描く。定義は「計算仕様」を参照。終点を大きく描く。`impute_strategy="drop"`のrunで欠損を含むファイルは軌跡を描けないため、その旨を表示する。

### 7. T² / Q

- run は transform 画面で選んだ表示する run を使い、表示ファイルの選択は使わない。
- 選択項目：色分けの列、管理図の並び順の列。
  - 選択は`/monitoring?color=…&order=…`のクエリで表す。
  - 色分けは画面名のカードに置き、3つの図に共通とする。並び順は Control charts のカードに置く。カードは上から、画面名・Control charts・T² × Q scatter・Selected points とする。
- 色分けはスコア散布図と同じく、既定は`ui.default_color_by`とし、「none」も選べる。数値の列は連続カラースケール（3つの図とも、UCL内外の系列で1つのスケールを共有し、カラーバーは1本だけ描く。全件欠損の列は灰色で描き、カラーバーを出さない）、それ以外の列は値ごとの系列（欠損は1つの系列）とする。色分けしたときのUCL超過点は、値の色のまま大きいひし形にして赤い縁で強調する。色分けしないときは従来どおりUCL超過点を赤いひし形で描く。
- 1点が1ファイル（`scores.parquet`の`mahalanobis_*`・`spe_*`列。`impute_strategy="drop"`で除いたファイルは含まず、その旨を表示する）。列名とUCLはrunの`config.json`の`MahalanobisConfig`・`SpeConfig`に従う。
- ファイル順の管理図（T²とQ、UCL線付き）。横軸はファイルの順位（1..N）とする。順序は`ui.default_order_by`の列、または画面で選んだ列の昇順（欠損は末尾、同じ値はファイル名の自然順）とし、「file name (natural order)」ではファイル名の自然順とする。UCL超過点を強調し、超過したファイルの件数と一覧を表示する。
- T² × Q散布図（UCL線付き）。いずれかのUCLを超えた点を強調する。
- 3つの図の点のクリックと範囲選択（box / lasso）で、選んだ点を1つの「選んだ点の表」に表示する。表の列は、ファイル名・メタデータ列・T²・Q・それぞれのUCL超過とする。点をクリックしても画面は移らず、表示ファイルにも追加しない。
- 描画関数は`visualize/monitoring.py`（`create_control_chart`・`create_t2_q_scatter`）に置く。

### 選んだ点の表（スコア・T² / Q）

- 画面ごとに1つ、図の下に Selected points のカードとして置く（`partials/point_table.html`）。行は`services/point_table.py::point_table`が作り、全ファイルの行（`format_value`で整形した文字列。先頭はstem）をページにJSONで埋め込む。`app.js`が選んだ点の行を図の並び順で表示する。ファイル数1万・15列でおよそ1.7 MBになる。
- クリックではその1行、範囲選択では囲んだすべての点の行を表示し、毎回前の表示を置き換える。選択の解除（ダブルクリック）で表を空にする。
- 図のモードバーに Box Select と Lasso Select を出す（`layout.modebar.add`）。
- 表は高さに上限を設け、行が多いときは表の中でスクロールする。

## 計算仕様

以下は`feature_engineering/`（または`visualize/`のデータ準備関数）に独立した関数として追加し、単体テストする。

- 第k成分のみの寄与：前処理後の空間で$t_k \mathbf{w}_k$を求め、スケーリングの逆変換のうち中心化を除く部分（scaleの乗算）だけを適用して元のスケールに戻す。
- 部分スコア軌跡：1ファイルの前処理済み行$\mathbf{x}$に対し、fit時と同じ規則で補完・外れ値処理・スケーリングを適用して中心化した$\mathbf{z}$を得る（`impute_model`→`outlier_model`→`scaling_model`の`apply`をこの順に呼び、`pca.mean_`を引く。`scaling_strategy="none"`では中心化をPCAに任せているため）。特徴量を`(Step, Sequence, StepTime)`の昇順に並べ（`(Step, Sequence)`は実際の時間順でなくStep・Sequenceの値の順とする）、その順の累積和$\tau \mapsto \sum_{f \le \tau} z_f w_{m,f}$（成分nも同様）を求める。各`(Step, Sequence, StepTime)`点を1点とする。終点は通常のスコアと一致する（`whiten=False`の場合）。累積和は`float64`で計算する。
- Q寄与：1ファイルの前処理済み行に、fit時と同じ補完・外れ値処理・スケーリングを適用した$\mathbf{x}$（`pca.transform`へ渡す値）と、その再構成$\hat{\mathbf{x}}$との特徴量ごとの差の二乗$(x_f - \hat{x}_f)^2$とする（SPEC.md「Q統計量（SPE）仕様」と同じ空間）。再構成に使う成分数は、そのrunの`SpeConfig.cumulative_explained_variance`が選ぶ成分数に固定し、探索画面で選んだkは使わない。これにより、全特徴量（ビニング前）の寄与の総和は`scores.parquet`のQと一致する。`prepare_rows`の補完・外れ値処理後の値を`scale_rows`（`scaling_model.apply`と同じ値を `model.feature_arrays` の配列で計算する）でスケーリングし、再構成は`reconstruct_standardized`で求めてスケーリングの逆変換はしない（`pca/q_contribution.py::q_contribution`）。
  - `X.npy`・`components.npy`を`float32`で保存したrunでは、丸めで総和がQとずれることがある（大きなベースラインに小さな変動が乗るデータなど）。表示するファイルごとに、ビニング前の全特徴量の寄与の総和と`scores.parquet`のQ（`SpeConfig.spe_column`）を比べ、相対差（|総和 − Q| / |Q|）が1%を超えたら、両方の値と「`jobs.artifact_dtype = "float64"`で再実行する」旨（Run the fit again with jobs.artifact_dtype = "float64".）を警告として表示する（`services/q_consistency.py`）。
- ローディングの波長集計：`reshape_pca_components`の結果（`component`・`wavelength`・`coefficient`列を持つlong形式）を成分と波長でgroup_byし、平均・RMS・絶対値平均のいずれかを求める（`aggregate_loadings_by_wavelength`）。モデル画面では、選んだ2成分だけを`features.parquet`と`components.npy`から同じlong形式にして渡す。
- ヒートマップのビニング：時間方向を等間隔のビンに分け、ビン内平均をとる。波長方向は間引かない（1200列程度を想定）。
- 実行前のメモリ見積もり：catalogの行数・波長数と前処理設定から特徴量数$F$を求め、$N \times F \times 8$Bの係数倍を表示する。
  - $F$は、`(Step, Sequence)`ごとにファイル間で最大の行数を`target_steps`・`edge_trim`（`StepTime`が等間隔と仮定）・時間方向の間引きで減らした時刻数と、波長数が最大のファイルの波長を`wavelength_range`・波長方向の間引きで減らした数の積とする。sparse列除去前の値である。
  - 係数はfitの経路で分ける。`run_fit`を合成データ（$N$=300・600、$F$=30,000）で実測したピークメモリから、$N$=600の値を切り上げた。$N$に比例しない分があるため、$N$が大きいほど実際の係数は小さくなる。
    - NumPyの高速経路（`impute_strategy="drop"`かつ`scaling_strategy="none"`）：10倍（実測 約9倍）
    - polarsの経路（上記以外の`"drop"`・`"median"`）：16倍（実測 約12〜16倍）
    - `"kmeans"`補完：24倍（実測 約23〜24倍）

## テスト方針

- `feature_engineering/`・`visualize/`の追加関数：単体テスト。
- `webui/services/`・`webui/jobs/`：小さなParquet・ダミーCSV・settings.tomlを`tests/`に用意して単体テストする。ジョブ関数はサブプロセスを介さずに直接呼ぶ。
- 実行器：実際に子プロセスを起動するテストで、次を確認する。
  - 実行中のジョブをキャンセルした後、待機中のジョブと新しく投入したジョブが成功する。
  - 待機中のジョブをキャンセルしても、実行中のジョブが成功する。
  - 子プロセスが異常終了したrunが`failed`になり、次のジョブが実行される。
- Q寄与：補完で欠損を埋めた行を含むデータで、ビニング前の寄与の総和が`scores.parquet`のQと一致する。
- `webui/routes/`：FastAPIの`TestClient`で、ステータスコードと返すpartialの主要要素を確認する。
- ブラウザ上の操作（クリック・スライダー等）は、`e2e` markerを付けたPlaywrightのテスト（`tests/webui/e2e/`）で主要な操作のみ確認する。

## マイルストーン

1. 基盤：settings、Workspace、DuckDB、catalogジョブ、データ選択画面
2. fitジョブ：前処理・PCA画面、run成果物、進捗・キャンセル、run一覧
3. スペクトル探索：元データ・前処理済みのヒートマップ、トレンド、スライダー
4. 再構成系：単一成分・累積再構成・残差の表示
5. スコア・部分スコア軌跡
6. T²/Q：管理図・散布図・Q寄与ヒートマップ
7. モデル：寄与率・ローディング・PCA成分のヒートマップを、ファイルを見る画面から分けて1画面にまとめる

## 未決事項

- float32保存の影響の実データでの確認。比較スクリプト`uv run -m flat_pca.webui.compare_artifact_dtypes --run-dir <runディレクトリ>`は、runの設定で1回だけ再fitし、その結果をfloat64とfloat32で保存して、それぞれから復元したモデルで計算したスコア・T²・Q・再構成の差を表示する（dtypeごとに再fitすると、randomized PCAの乱数による差が混ざるため）。合成データでの相対差は$10^{-7}$程度だった。
- メモリ見積もりの係数の実データでの確認（合成データの実測値で暫定的に決めた）。
