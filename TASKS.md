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

- Status: pending
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

- Status: pending
- Priority: 6
- Depends on: TASK-001

### Requirements

- 各ファイルを`filename`とflatten特徴量を持つ1行へ変換する。
- 特徴量を数値の`(w, s, q, t)`で昇順に並べる。
- `Sequence`を`q`として`f"{w}*{s}*{q}_{t}"`形式の列名を生成する。
- `s`および`q`を`int`型で表現する。
- `t`を`f"{t:.2f}nm"`で表現する。
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

- Status: pending
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

- Status: pending
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
