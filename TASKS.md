# Tasks

このファイルは`SPEC.md`を実装するRalph loopのタスク台帳である。各loopは依存関係を満たした`pending`タスクを1件だけ処理し、要件と検証条件をすべて満たした場合だけ`Status`を`completed`へ変更する。

## TASK-001: Parquet入力の読み込みと検証

- Status: completed
- Priority: 1
- Depends on: none

### Requirements

- 1件以上のパスを受け取り、`Path(path).resolve()`で正規化した絶対パスの文字列表現で昇順に処理する。
- 存在しないパスを`FileNotFoundError`で拒否する。
- 重複する`path.stem`を`ValueError`で拒否する。
- `Time`、`Step`、`Sequence`を必須の数値列として検証する。
- `f"{v:.1f}nm"`形式の波長列と数値型のスペクトル強度を検証する。
- null、NaN、無限大および重複した`(Time, Step, Sequence)`を拒否する。
- 全ファイルの波長集合と`(Time, Step, Sequence)`集合が一致することを検証する。

### Tests

- `tests/fixtures/real_subset`の1件および複数件を正常入力として読み込める。
- 実フィクスチャから作った変種で、必須列欠落、不正な波長名、不正値、重複キー、ファイル間不整合を検出できる。
- 入力順を反転しても処理順が変わらない。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-002: t方向smoothing

- Status: completed
- Priority: 2
- Depends on: TASK-001

### Requirements

- `t_smoothing_window=None`では入力を変更しない。
- 有限かつ0より大きい片側窓幅だけを受け付ける。
- `(Step, Sequence, w)`ごとに実Timeの閉区間を使って算術平均する。
- 不等間隔Time、区間端、グループ分離を`SPEC.md`どおり扱う。
- メタデータ、行数および波長列数を保持する。

### Tests

- `tests/fixtures/real_subset`に、隣接時点を含む窓と含まない窓を適用して期待値を検証する。
- 実フィクスチャから作った変種で、異なる`Step`と`Sequence`が混ざらないことを検証する。
- `None`、0、負数、NaNおよび無限大の挙動を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k t_smoothing`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-003: w方向smoothing

- Status: completed
- Priority: 3
- Depends on: TASK-001

### Requirements

- `w_smoothing_window=None`では入力を変更しない。
- 有限かつ0より大きい片側窓幅だけを受け付ける。
- `(Time, Step, Sequence)`ごとに実波長の閉区間を使って算術平均する。
- 不等間隔波長、波長域の端、グループ分離を`SPEC.md`どおり扱う。
- メタデータ、行数および波長列数を保持する。

### Tests

- `tests/fixtures/real_subset`に、隣接波長を含む窓と含まない窓を適用して期待値を検証する。
- 実フィクスチャから作った変種で、異なるメタデータ行が混ざらないことを検証する。
- `None`、0、負数、NaNおよび無限大の挙動を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k w_smoothing`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-004: t方向規格化

- Status: completed
- Priority: 4
- Depends on: TASK-001

### Requirements

- `t_normalization_range=None`では入力を変更しない。
- `(Step, Sequence, w)`ごとに指定したTime閉区間の算術平均で全Timeの値を割る。
- 区間端を含め、異なるグループを分離する。
- 不正な範囲、空区間、0または有限でない除数を`ValueError`で拒否する。

### Tests

- `tests/fixtures/real_subset`の実Time区間を使い、基準区間の平均が1になることを検証する。
- 実フィクスチャから作った変種でグループ分離と各エラー条件を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k t_normalization`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-005: w方向規格化

- Status: completed
- Priority: 5
- Depends on: TASK-001

### Requirements

- `w_normalization_range=None`では入力を変更しない。
- `(Time, Step, Sequence)`ごとに指定した波長閉区間の算術平均で全波長の値を割る。
- 区間端を含め、異なるメタデータ行を分離する。
- 不正な範囲、空区間、0または有限でない除数を`ValueError`で拒否する。

### Tests

- `tests/fixtures/real_subset`の実波長区間を使い、基準区間の平均が1になることを検証する。
- 実フィクスチャから作った変種でグループ分離と各エラー条件を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k w_normalization`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-006: 決定的なflatten処理

- Status: completed
- Priority: 6
- Depends on: TASK-001

### Requirements

- 各ファイルを`filename`とflatten特徴量を持つ1行へ変換する。
- 特徴量を数値の`(w, s, q, t)`で昇順に並べる。
- `Sequence`を`q`として`f"{w}_{s}_{q}_{t}"`形式の列名を生成する。
- `s`および`q`を`int`型で表現する。
- `t`を`f"{t:.2f}"`で表現する。
- 生成後の列名重複を`ValueError`で拒否する。
- 複数ファイルの結果を入力パス順に依存しない行順で結合する。

### Tests

- `tests/fixtures/real_subset`の1件で列名、列順、値、行数を検証する。
- 同フィクスチャの複数件で`filename`、行順、入力順非依存性を検証する。
- 実フィクスチャから作った変種で複数`Step`・`Sequence`と列名衝突を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k flatten`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-007: 前処理とPCAを統合した公開API

- Status: completed
- Priority: 7
- Depends on: TASK-002, TASK-003, TASK-004, TASK-005, TASK-006

### Requirements

- `SPEC.md`のシグネチャを持つ公開関数`flatten_pca`を実装する。
- 前処理をt smoothing、w smoothing、t規格化、w規格化の順に適用する。
- `fit_and_transform_pca`を`scaling_strategy="none"`、`max_n_component=None`、`impute_strategy="drop"`、`outlier_strategy=None`で呼び出す。
- flatten特徴量を保持し、`pca-1`から始まるPCAスコア列を追加する。
- `n_component`を検証し、`pl.DataFrame`を返す。
- 公開APIを`spca.feature_engineering`からimportできるようにする。

### Tests

- `tests/fixtures/real_subset`全6件を用いて、前処理なしのend-to-end結果を検証する。
- 同フィクスチャで各前処理単独および全前処理併用時の適用順を検証する。
- 出力の行数、列順、`filename`、PCA列数、有限値を検証する。
- PCA結果は符号反転を許容し、分散、部分空間または再構成結果で検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py -k flatten_pca`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-008: 仕様全体の回帰確認と利用情報

- Status: completed
- Priority: 8
- Depends on: TASK-007

### Requirements

- `SPEC.md`の全要件と実装・テストの対応を確認する。
- 公開関数と公開クラスに型ヒントとNumPy形式のdocstringを付ける。
- READMEにPython APIの最小利用例、入力スキーマ、前処理順、出力形式を記載する。
- 既存の可視化およびPCA関連機能を壊さない。

### Tests

- `tests/fixtures/real_subset`全6件を使うend-to-end回帰テストを実行する。
- 公開importと最小利用例が実行可能であることを検証する。
- 全テストとlintが成功することを確認する。

### Acceptance commands

- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-009: Flatten-PCA実装の責務分割

- Status: completed
- Priority: 9
- Depends on: TASK-008

### Requirements

- `flatten_pca`の公開シグネチャ、戻り値、例外、処理順および
  `from spca.feature_engineering import flatten_pca`という公開importを変更しない。
- 現在の`flatten_pca.py`が持つ以下の責務を、凝集度の高いモジュールへ分割する。
  - 公開APIと処理のオーケストレーション
  - Parquet入力の読み込みと検証
  - t/w方向smoothing
  - t/w方向normalization
  - deterministic flatten処理
- 必要に応じて`spca.feature_engineering.flatten_pca`をパッケージ化する。
- パッケージの`__init__.py`は公開APIのre-exportだけを行う薄いfacadeとする。
- 下位モジュールから公開API・オーケストレーション層への逆依存を作らない。
- 循環importおよび責務の曖昧な`utils.py`を作らない。
- 既存のprivate helperは、それを所有する責務別モジュールへ移動する。
  private helperの旧importパス互換性は要求しない。
- テストも責務単位に分割し、共通する実フィクスチャ設定は
  `conftest.py`などへ集約する。
- 既存テストの検証内容を削除または弱体化しない。
- すべての関数とクラスに型ヒントとNumPy形式docstringを付ける。
- 未使用コード、不要な互換レイヤー、一時ファイルを残さない。

### Tests

- `tests/fixtures/real_subset`を直接読み込むend-to-endテストを追加または更新する。
- 公開importと公開APIのシグネチャが維持されることを検証する。
- 前処理なし、各前処理単独、全前処理併用の既存結果が維持されることを検証する。
- 入力検証、smoothing、normalization、flatten、PCA統合を、それぞれの
  責務に対応するテストモジュールで検証する。
- 入力順に依存しない行順・列順と、PCA結果の既存契約を維持する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-010: 空の入力パスの明示的な検証

- Status: completed
- Priority: 10
- Depends on: TASK-009

### Requirements

- `flatten_pca`へ空の入力パス列を渡した場合、処理途中の内部例外ではなく、入力が1件以上必要であることを示す明確な`ValueError`を送出する。
- リスト、タプルおよびジェネレータなど、既存の反復可能な入力に対する挙動を維持する。
- 公開APIのシグネチャ、戻り値、前処理順および公開importパスを変更しない。
- すべての関数とクラスの型ヒントおよびNumPy形式docstringを維持する。

### Tests

- 空のリスト、空のタプルおよび空のジェネレータが`ValueError`になることを検証する。
- `tests/fixtures/real_subset`のParquetファイルを直接読み込み、1件のリストおよびジェネレータ入力が従来どおり処理できることを検証する。
- 既存の入力検証テストを弱体化しない。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca_input.py`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-011: t方向の間引き

- Status: completed
- Priority: 11
- Depends on: TASK-010

### Requirements

- 間引き専用の責務を持つモジュールに、`apply_t_downsampling`を実装する。
- 検証済みの全入力ファイルの`Time`列から、重複を除いて数値昇順に並べた`list[float]`を一度だけ生成する。
- `apply_t_downsampling`は、対象フレーム、t方向のUnique配列および間引き間隔を引数として受け取る。
- Unique配列のindex `0, stride, 2 * stride, ...`にある`Time`値と一致する行を、`Step`および`Sequence`にかかわらずすべて残す。
- 選択indexに該当しない末尾の値を追加で残さない。
- 間引き間隔`1`ではすべての行を残す。
- boolではない1以上の整数だけを間引き間隔として受け付け、それ以外は`ValueError`で拒否する。
- すべての関数に型ヒントとNumPy形式のdocstringを付ける。

### Tests

- `tests/fixtures/real_subset`のParquetを直接読み込み、全入力から昇順のUnique配列を生成できることを検証する。
- 実フィクスチャに間引き間隔`1`および`2`を適用し、期待する`Time`と全`Step`・`Sequence`の行だけが残ることを検証する。
- 実フィクスチャから`pl.DataFrame`を導出し、末尾が選択indexに該当しない場合に末尾が残らないことを検証する。
- 0、負数、boolおよび非整数の間引き間隔を拒否することを検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca_downsampling.py -k t_downsampling`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-012: w方向の間引き

- Status: completed
- Priority: 12
- Depends on: TASK-011

### Requirements

- 間引き専用モジュールに、`apply_w_downsampling`を実装する。
- 検証済みの全入力ファイルに共通する波長列名を数値へ変換し、重複を除いて昇順に並べた`list[float]`を一度だけ生成する。
- `apply_w_downsampling`は、対象フレーム、w方向のUnique配列および間引き間隔を引数として受け取る。
- Unique配列のindex `0, stride, 2 * stride, ...`にある波長値に対応する波長列だけを残す。
- 選択indexに該当しない末尾の波長列を追加で残さない。
- `Time`、`Step`および`Sequence`列を常に保持する。
- 間引き間隔`1`ではすべての波長列を残す。
- boolではない1以上の整数だけを間引き間隔として受け付け、それ以外は`ValueError`で拒否する。
- すべての関数に型ヒントとNumPy形式のdocstringを付ける。

### Tests

- `tests/fixtures/real_subset`のParquetを直接読み込み、全入力から数値昇順の波長Unique配列を生成できることを検証する。
- 実フィクスチャに間引き間隔`1`および`2`を適用し、期待する波長列と全メタデータ列だけが残ることを検証する。
- 実フィクスチャから`pl.DataFrame`を導出し、末尾が選択indexに該当しない場合に末尾の波長列が残らないことを検証する。
- 0、負数、boolおよび非整数の間引き間隔を拒否することを検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca_downsampling.py -k w_downsampling`
- `uv run -m pytest`
- `uv run -m ruff check .`

## TASK-013: 間引きとFlatten-PCA公開APIの統合

- Status: pending
- Priority: 13
- Depends on: TASK-011, TASK-012

### Requirements

- `flatten_pca`の公開APIに、デフォルト値`1`の`t_downsampling_stride`および`w_downsampling_stride`を追加する。
- Unique配列は公開APIの引数にせず、入力検証後に全入力から内部生成する。
- t smoothing、w smoothing、t規格化、w規格化、t間引き、w間引き、flattenの順に処理する。
- 全入力ファイルへ同じt/w方向のUnique配列と間引き間隔を適用し、入力間で同じflatten特徴量集合を維持する。
- PCAへ渡す特徴量と`n_component`の上限を、間引き後のflatten特徴量に基づいて決定する。
- 間引き間隔`1`の既定動作で、従来のflatten結果およびPCA結果を維持する。
- READMEの公開API例、引数説明および処理順を新しい間引き機能に合わせて更新する。
- 公開importパス、戻り値、既存の前処理引数および入力順に依存しない出力順を維持する。

### Tests

- `tests/fixtures/real_subset`の複数Parquetを直接使用し、t方向のみ、w方向のみ、および両方向を間引いたend-to-end結果を検証する。
- smoothingと規格化が間引き前の全観測値を使用すること、および仕様どおりの処理順を検証する。
- 間引き後のflatten列名、列順、特徴量数、PCA列数、有限値および入力順非依存性を検証する。
- 間引き間隔`1`で既存のend-to-end結果が変わらないことを検証する。
- 公開APIから不正な間引き間隔を渡した場合の`ValueError`を検証する。

### Acceptance commands

- `uv run -m pytest tests/feature_engineering/test_flatten_pca_downsampling.py`
- `uv run -m pytest tests/feature_engineering/test_flatten_pca.py`
- `uv run -m pytest`
- `uv run -m ruff check .`
